/**********************************************************************
 *  bigram_streamer_opt.c
 *  ------------------------------------------------------------------
 *  • Uses 64-bit xoroshiro128+ RNG
 *  • O(1) alias-table sampling for unigrams and bigrams
 *  • Multithreaded; no frequent atomic increments
 *  • Temperature baked in
 *********************************************************************/

#define _GNU_SOURCE
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ---------- special tokens ---------------------------------------- */
#define BOS_TOKEN "<s>"
#define EOS_TOKEN "</s>"
#define UNK_TOKEN "<unk>"

/* ---------- filtering --------------------------------------------- */
static const char *FILTER_SKIP[] = {UNK_TOKEN, NULL};
static const char *FILTER_HIDE[] = {NULL};

/* ---------- limits ------------------------------------------------ */
#define MAX_SENT_LEN 64
#define BUF_SIZE 4096
#define SAFE_LOOP 10000

/* ---------- RNG: xoroshiro128+ ------------------------------------ */
typedef struct {
    uint64_t s[2];
} rng_t;

// Rotate left
static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// Next 64-bit from xoroshiro128+
static inline uint64_t rng_next64(rng_t *r) {
    uint64_t s0 = r->s[0];
    uint64_t s1 = r->s[1];
    uint64_t res = s0 + s1;
    r->s[1] ^= s0;
    r->s[0] = rotl(s0, 24) ^ r->s[1] ^ (r->s[1] << 16);
    r->s[1] = rotl(r->s[1], 37);
    return res;
}

// Jump function for rng skip-ahead
static void rng_jump(rng_t *r) {
    static const uint64_t J[2] = {0x8764000bafcaULL, 0xf542d2d3e851ULL};
    uint64_t s0 = 0, s1 = 0;
    for (int b = 0; b < 2; b++) {
        for (int i = 0; i < 64; i++) {
            if (J[b] & (1ULL << i)) {
                s0 ^= r->s[0];
                s1 ^= r->s[1];
            }
            rng_next64(r);
        }
    }
    r->s[0] = s0;
    r->s[1] = s1;
}

// Seed from /dev/urandom or fallback
static rng_t secure_seed(void) {
    rng_t R;
    FILE *f = fopen("/dev/urandom", "rb");
    if (!f || fread(&R, sizeof R, 1, f) != 1) {
        // Fallback: time-based
        uint64_t t = (uint64_t)time(NULL);
        R.s[0] = t;
        R.s[1] = t ^ 0xdeadbeefcafebabeULL;
    }
    if (f) fclose(f);
    if (!R.s[0] && !R.s[1]) {
        // ensure non-zero
        R.s[1] = 1;
    }
    return R;
}

// Generate double in [0,1)
static inline double rng_uniform_f64(rng_t *r) {
    // 53 bits for double fraction
    uint64_t x = rng_next64(r);
    x >>= 11; // keep the low 53 bits
    return (double)x * (1.0 / (double)(1ULL << 53));
}

/* ---------- ALIAS TABLE for O(1) sampling --------------------------
   We'll store each distribution in an alias structure:

   struct alias_table {
       int     n;
       int    *alias;  // for each slot
       double *cut;    // probability cut boundary
       int    *ids;    // actual token IDs (unigram or bigram next)
   };

   Then sampling:
       double r = rng_uniform_f64(...) * n;
       int i = (int) r;
       double frac = r - i;
       if (frac < cut[i]) index = i; else index = alias[i];
       real token = ids[index];
 -------------------------------------------------------------------*/
typedef struct {
    int n;
    int   *alias;    // length n
    double *cut;     // length n
    int   *ids;      // length n  (the actual token for that slot)
} alias_table_t;

// Build alias table from an array of probabilities
// 'ids' has the same length as 'prob' so we can keep track of which token is which
static alias_table_t build_alias_table(const double *prob, const int *ids, int n) {
    alias_table_t tbl;
    tbl.n = n;
    tbl.alias = (int*)malloc(n * sizeof(int));
    tbl.cut   = (double*)malloc(n * sizeof(double));
    tbl.ids   = (int*)malloc(n * sizeof(int));

    if (!tbl.alias || !tbl.cut || !tbl.ids) {
        fprintf(stderr, "malloc failed in build_alias_table\n");
        exit(1);
    }

    // create temporary arrays
    int *small = (int*)malloc(n * sizeof(int));
    int *large = (int*)malloc(n * sizeof(int));
    if (!small || !large) {
        fprintf(stderr, "malloc failed in build_alias_table (temp)\n");
        exit(1);
    }

    double *p = (double*)malloc(n * sizeof(double));
    if (!p) {
        fprintf(stderr, "malloc failed in build_alias_table (p)\n");
        exit(1);
    }

    // sum probabilities
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += prob[i];
    }
    if (sum <= 0.0) {
        // fallback: uniform distribution if somehow all probs are zero
        for (int i = 0; i < n; i++) {
            p[i] = 1.0 / (double)n;
        }
    } else {
        // normalize
        for (int i = 0; i < n; i++) {
            p[i] = (prob[i] / sum) * n;
        }
    }

    // copy ids
    for (int i = 0; i < n; i++) {
        tbl.ids[i] = ids[i];
    }

    // fill small/large
    int small_count = 0;
    int large_count = 0;
    for (int i = 0; i < n; i++) {
        if (p[i] < 1.0) small[small_count++] = i;
        else            large[large_count++] = i;
    }

    while (small_count > 0 && large_count > 0) {
        int s = small[--small_count];
        int l = large[--large_count];
        tbl.cut[s]   = p[s];   // cut boundary
        tbl.alias[s] = l;      // alias points to 'l'
        p[l] = (p[l] + p[s]) - 1.0; // reduce p[l]
        if (p[l] < 1.0) {
            small[small_count++] = l;
        } else {
            large[large_count++] = l;
        }
    }

    // remaining
    while (large_count > 0) {
        int l = large[--large_count];
        tbl.cut[l]   = 1.0;
        tbl.alias[l] = l;
    }
    while (small_count > 0) {
        int s = small[--small_count];
        tbl.cut[s]   = 1.0;
        tbl.alias[s] = s;
    }

    free(small);
    free(large);
    free(p);
    return tbl;
}

// Sample from alias table
static inline int alias_sample(rng_t *r, const alias_table_t *tbl) {
    // note: n can be zero if there's no valid distribution
    if (tbl->n == 0) return -1; // no valid token
    double x = rng_uniform_f64(r) * (double)tbl->n;
    int i = (int)x;
    double frac = x - (double)i;
    if (frac < tbl->cut[i]) {
        return tbl->ids[i];
    } else {
        return tbl->ids[tbl->alias[i]];
    }
}

/* ---------- structures for unigrams and bigrams ------------------- */

// For unigrams
typedef struct {
    char *w; // the token text
} unig_t;

static unig_t *UNI = NULL;     // array of size V
static int V = 0;              // number of unigrams (filtered)
static int BOS = -1, EOS = -1; // IDs for <s> and </s>
static double INV_TEMP = 1.0;  // 1/temperature

// ALIAS table for unigrams
static alias_table_t UNI_ALIAS;

// For bigrams, store an array of (next, prob)
typedef struct {
    int next;
    double p;
} bigr_entry_t;

typedef struct {
    int n;
    bigr_entry_t *list; // [n]
    alias_table_t alias; // built later
} bigr_bucket_t;

// We'll allocate bigr_bucket_t for each possible 'prev' ID
static bigr_bucket_t *BIGRAMS = NULL; // size = V

/* ---------- CLI options ------------------------------------------- */
static const char *opt_model = NULL;
static int opt_min = 5, opt_max = 20;
static double opt_temp = 1.0;
static int opt_thr = 0;
static long opt_bench = 0;
enum { MODE_REGULAR, MODE_TOKENIZER } opt_mode = MODE_REGULAR;

/* ---------- small helpers ----------------------------------------- */
static int str_in(const char *word, const char *const *list) {
    for (; *list; ++list) {
        if (strcmp(*list, word) == 0) return 1;
    }
    return 0;
}

// We store all unigrams in UNI[]. Return the ID by string match
static int id_of(const char *w) {
    // naive linear search: if V is large, you might want a dictionary
    for (int i = 0; i < V; i++) {
        if (strcmp(UNI[i].w, w) == 0) return i;
    }
    return -1;
}

/* ---------- read ARPA & build data -------------------------------- */
static void load_arpa(const char *fn) {
    FILE *fp = fopen(fn, "r");
    if (!fp) {
        perror(fn);
        exit(1);
    }
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    int sec = 0;

    // We'll store raw unigrams in arrays first
    // (We only know how many after reading them; dynamic reallocation is okay.)
    // We'll keep track of their probability (before alias build).
    double *uni_prob = NULL;

    // 1) parse 1-grams
    while ((len = getline(&line, &cap, fp)) != -1) {
        if (line[0] == '\\') {
            if (!strncmp(line, "\\1-grams:", 9)) {
                sec = 1;
                continue;
            }
            if (!strncmp(line, "\\2-grams:", 9)) break;
            continue;
        }
        if (sec != 1) continue;

        char lp[64], w[1024];
        if (sscanf(line, "%63s %1023s", lp, w) != 2) continue;
        if (str_in(w, FILTER_SKIP)) continue; // skip <unk> etc

        long double logp = strtold(lp, NULL);
        // pow(10, logp) then pow(..., 1/temp)
        // => pow(10^(logp), 1/temp) = 10^(logp / temp)
        long double p0 = powl(10.0L, logp);
        p0 = powl(p0, (long double)INV_TEMP);

        // add new unigram
        UNI = realloc(UNI, (V + 1) * sizeof(*UNI));
        uni_prob = realloc(uni_prob, (V + 1) * sizeof(*uni_prob));
        if (!UNI || !uni_prob) {
            fputs("Out of memory loading unigrams\n", stderr);
            exit(1);
        }
        UNI[V].w = strdup(w);
        uni_prob[V] = (double)p0;
        V++;
    }
    if (!V) {
        fputs("No unigrams found!\n", stderr);
        exit(1);
    }

    // Identify BOS/EOS
    for (int i = 0; i < V; i++) {
        if (strcmp(UNI[i].w, BOS_TOKEN) == 0) BOS = i;
        if (strcmp(UNI[i].w, EOS_TOKEN) == 0) EOS = i;
    }
    if (BOS < 0) fprintf(stderr, "Warning: no <s> found\n");
    if (EOS < 0) fprintf(stderr, "Warning: no </s> found\n");

    // 2) allocate bigram buckets
    BIGRAMS = (bigr_bucket_t*)calloc(V, sizeof(bigr_bucket_t));
    if (!BIGRAMS) {
        fputs("Out of memory for BIGRAMS\n", stderr);
        exit(1);
    }

    // 3) parse 2-grams
    rewind(fp);
    sec = 0;
    while ((len = getline(&line, &cap, fp)) != -1) {
        if (line[0] == '\\') {
            if (!strncmp(line, "\\2-grams:", 9)) {
                sec = 2;
                continue;
            }
            continue;
        }
        if (sec != 2) continue;

        char lp[64], w1[1024], w2[1024];
        if (sscanf(line, "%63s %1023s %1023s", lp, w1, w2) != 3) continue;
        if (str_in(w1, FILTER_SKIP) || str_in(w2, FILTER_SKIP)) continue;

        int id1 = id_of(w1);
        int id2 = id_of(w2);
        if (id1 < 0 || id2 < 0) continue;

        // probability
        long double logp = strtold(lp, NULL);
        long double p0 = powl(10.0L, logp);
        p0 = powl(p0, (long double)INV_TEMP);

        // append to bigram bucket
        bigr_bucket_t *bk = &BIGRAMS[id1];
        bk->list = realloc(bk->list, (bk->n + 1) * sizeof(bigr_entry_t));
        if (!bk->list) {
            fputs("Out of memory adding bigram\n", stderr);
            exit(1);
        }
        bk->list[bk->n].next = id2;
        bk->list[bk->n].p    = (double)p0;
        bk->n++;
    }
    free(line);
    fclose(fp);

    // 4) build alias table for unigrams
    //    First gather unigrams' IDs in an array
    int *uni_ids = (int*)malloc(V * sizeof(int));
    if (!uni_ids) {
        fputs("Out of memory building unigrams\n", stderr);
        exit(1);
    }
    for (int i = 0; i < V; i++) {
        uni_ids[i] = i;
    }
    // Build alias for unigrams
    UNI_ALIAS = build_alias_table(uni_prob, uni_ids, V);

    free(uni_ids);
    free(uni_prob);

    // 5) build alias tables for bigrams
    for (int i = 0; i < V; i++) {
        bigr_bucket_t *bk = &BIGRAMS[i];
        if (bk->n == 0) {
            // empty bucket => no bigrams from i
            bk->alias.n = 0;
            bk->alias.alias = NULL;
            bk->alias.cut   = NULL;
            bk->alias.ids   = NULL;
            continue;
        }
        // gather prob[] and ids[]
        double *prob = (double*)malloc(bk->n * sizeof(double));
        int    *ids  = (int*)malloc(bk->n * sizeof(int));
        if (!prob || !ids) {
            fputs("Out of memory building bigram alias\n", stderr);
            exit(1);
        }
        for (int j = 0; j < bk->n; j++) {
            prob[j] = bk->list[j].p;
            ids[j]  = bk->list[j].next;
        }
        bk->alias = build_alias_table(prob, ids, bk->n);
        free(prob);
        free(ids);
    }
}

// Sample from unigrams, skipping BOS
static inline int draw_uni(rng_t *R) {
    while (1) {
        int w = alias_sample(R, &UNI_ALIAS);
        if (w < 0) {
            // fallback if something weird
            return 0;
        }
        if (w == BOS) continue; // skip BOS in generation
        return w;
    }
}

// Sample from bigrams if available, else fallback to unigrams
static inline int draw_bi(rng_t *R, int prev) {
    bigr_bucket_t *bk = &BIGRAMS[prev];
    if (bk->alias.n == 0) {
        return draw_uni(R);
    }
    while (1) {
        int w = alias_sample(R, &bk->alias);
        if (w < 0) return draw_uni(R); // fallback
        if (w == BOS) continue;        // skip BOS
        return w;
    }
}

/* ---------- worker thread ----------------------------------------- */
typedef struct {
    long start, end;
    int print;
    // local counters
    long local_tok;
    long local_sent;
} warg_t;

static pthread_mutex_t M_OUT = PTHREAD_MUTEX_INITIALIZER;

static void *worker(void *argp) {
    warg_t *A = (warg_t*)argp;
    rng_t R = secure_seed();

    // skip ahead for a distinct RNG state
    for (long j = 0; j < A->start; j++) {
        rng_jump(&R);
    }

    char buf[BUF_SIZE];
    A->local_tok = 0;
    A->local_sent = 0;

    for (long s = A->start; A->end < 0 || s < A->end; s++) {
        int prev = BOS, out = 0, iter = 0;
        size_t idx = 0;

        while (out < opt_max && iter++ < SAFE_LOOP) {
            int w = draw_bi(&R, prev);
            if (w == EOS && out < opt_min) {
                // ensure min length
                continue;
            }
            if (w == EOS) {
                // end of sentence
                break;
            }
            const char *word = UNI[w].w;
            // skip "hidden" tokens from output
            if (str_in(word, FILTER_HIDE)) {
                prev = w;
                continue;
            }
            // output if requested
            if (A->print) {
                size_t L = strlen(word);
                // flush if not enough space
                if (idx + L + 2 >= BUF_SIZE) {
                    pthread_mutex_lock(&M_OUT);
                    fwrite(buf, 1, idx, stdout);
                    idx = 0;
                    pthread_mutex_unlock(&M_OUT);
                }
                memcpy(buf + idx, word, L);
                idx += L;
                if (opt_mode == MODE_REGULAR) {
                    buf[idx++] = ' ';
                }
            }
            out++;
            prev = w;
        }

        if (A->print) {
            if (idx == 0) {
                buf[idx++] = ' ';
            }
            if (opt_mode == MODE_REGULAR) {
                buf[idx - 1] = '\n'; // replace last space with newline
            } else {
                // tokenizer mode: replace U+2581 (xe2 96 81) with space
                buf[idx] = '\0';
                for (size_t i = 0; i + 2 < idx; i++) {
                    if ((unsigned char)buf[i]   == 0xe2 &&
                        (unsigned char)buf[i+1] == 0x96 &&
                        (unsigned char)buf[i+2] == 0x81) {
                        buf[i] = ' ';
                        memmove(buf + i + 1, buf + i + 3, idx - (i+3));
                        idx -= 2;
                    }
                }
                buf[idx++] = '\n';
            }
            pthread_mutex_lock(&M_OUT);
            fwrite(buf, 1, idx, stdout);
            pthread_mutex_unlock(&M_OUT);
        }

        A->local_tok += out;
        A->local_sent += 1;

        if (A->end >= 0 && A->local_sent >= (A->end - A->start)) {
            // we've generated our share
            break;
        }
    }

    return NULL;
}

/* ---------- util: get CPU cores ----------------------------------- */
static int cores(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0 ? (int)n : 1);
}

/* ---------- CLI --------------------------------------------------- */
static void cli(int ac, char **av) {
    for (int i = 1; i < ac; i++) {
        if (!strcmp(av[i], "--model") && i + 1 < ac) {
            opt_model = av[++i];
        } else if (!strcmp(av[i], "--min") && i + 1 < ac) {
            opt_min = atoi(av[++i]);
        } else if (!strcmp(av[i], "--max") && i + 1 < ac) {
            opt_max = atoi(av[++i]);
        } else if (!strcmp(av[i], "--temp") && i + 1 < ac) {
            opt_temp = strtod(av[++i], NULL);
        } else if (!strcmp(av[i], "--threads") && i + 1 < ac) {
            opt_thr = atoi(av[++i]);
        } else if (!strcmp(av[i], "--bench") && i + 1 < ac) {
            opt_bench = atol(av[++i]);
        } else if (!strcmp(av[i], "--mode") && i + 1 < ac) {
            const char *m = av[++i];
            if (!strcmp(m, "regular")) {
                opt_mode = MODE_REGULAR;
            } else if (!strcmp(m, "tokenizer")) {
                opt_mode = MODE_TOKENIZER;
            } else {
                fprintf(stderr, "bad mode\n");
                exit(1);
            }
        } else {
            fprintf(stderr,
                    "usage: %s --model file.arpa [--min N] [--max N] "
                    "[--temp T] [--threads K] [--bench S] "
                    "[--mode regular|tokenizer]\n",
                    av[0]);
            exit(1);
        }
    }
    if (!opt_model) {
        fputs("--model required\n", stderr);
        exit(1);
    }
    if (opt_thr <= 0) opt_thr = cores();
    if (opt_temp <= 0.0) opt_temp = 1.0;
    INV_TEMP = 1.0 / opt_temp;
}

/* ---------- main -------------------------------------------------- */
int main(int argc, char **argv) {
    cli(argc, argv);
    load_arpa(opt_model);

    pthread_t *tid = malloc(opt_thr * sizeof(*tid));
    warg_t *args = calloc(opt_thr, sizeof(*args));
    if (!tid || !args) {
        fputs("Out of memory allocating threads\n", stderr);
        exit(1);
    }

    if (opt_bench == 0) {
        // infinite streaming
        fprintf(stderr, "[info] streaming (%s) on %d thread%s …\n",
                opt_mode == MODE_REGULAR ? "regular" : "tokenizer",
                opt_thr, opt_thr > 1 ? "s" : "");

        // We'll just run threads that print forever (until killed)
        for (int i = 0; i < opt_thr; i++) {
            args[i].start = i;
            args[i].end   = -1;
            args[i].print = 1;
            pthread_create(&tid[i], NULL, worker, &args[i]);
        }
        for (int i = 0; i < opt_thr; i++) {
            pthread_join(tid[i], NULL);
        }

    } else {
        // benchmarking mode
        long S = opt_bench; // number of sentences total
        long per = (S + opt_thr - 1) / opt_thr;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        for (int i = 0; i < opt_thr; i++) {
            long st = i * per;
            long en = (i + 1) * per > S ? S : (i + 1) * per;
            args[i].start = st;
            args[i].end   = en;
            args[i].print = 0;
            pthread_create(&tid[i], NULL, worker, &args[i]);
        }

        long total_sent = 0, total_tok = 0;
        for (int i = 0; i < opt_thr; i++) {
            pthread_join(tid[i], NULL);
            total_sent += args[i].local_sent;
            total_tok  += args[i].local_tok;
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        long sec_s  = (t1.tv_sec - t0.tv_sec);
        long nsec_s = (t1.tv_nsec - t0.tv_nsec);
        long double sec = (long double)sec_s + (long double)nsec_s * 1e-9L;
        if (sec <= 0) sec = 1e-9L;

        fprintf(stderr,
            "done: %ld sentences  %ld tokens  •  %.3Lf s  →  %.0Lf sent/s  %.0Lf tok/s\n",
            total_sent, total_tok, sec,
            (long double)total_sent / sec,
            (long double)total_tok  / sec);
    }

    // free alias structures
    free(tid);
    free(args);
    free(UNI_ALIAS.alias);
    free(UNI_ALIAS.cut);
    free(UNI_ALIAS.ids);

    // free unigrams
    for (int i = 0; i < V; i++) {
        free(UNI[i].w);
    }
    free(UNI);

    // free bigrams
    for (int i = 0; i < V; i++) {
        free(BIGRAMS[i].list);
        free(BIGRAMS[i].alias.alias);
        free(BIGRAMS[i].alias.cut);
        free(BIGRAMS[i].alias.ids);
    }
    free(BIGRAMS);

    return 0;
}
