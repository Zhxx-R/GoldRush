#include <cstdint>
#include <linux/limits.h>
#include <math.h>
#include <random>
#include <utility> // for std::pair
#include <iostream>
#include <string>
#include <limits.h>
#include <vector>
#include <thread> // for std::this_thread::sleep_for
#include <chrono> // for std::chrono::seconds
#include <queue>
#include <cstring> // for memset
#include <unistd.h>
#include <cassert>
#include <immintrin.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>

using std::vector;
using std::queue;
using std::pair;

typedef std::vector<std::vector<int>> Map;
typedef std::pair<int, int> Position;
typedef std::pair<int, int> Direction;

bool npc_area[17][17];
int npc_appear_round = 0;

Direction directions[4] = {
    {-1, 0},
    {1, 0},
    {0, -1},
    {0, 1}
};

const int map_size = 17;
double reward[map_size][map_size];

template <typename T>
std::pair<T, T> operator+(const std::pair<T, T>& lhs, const std::pair<T, T>& rhs) {
    return std::make_pair(lhs.first + rhs.first, lhs.second + rhs.second);
}
template <typename T>
std::pair<T, T> operator-(const std::pair<T, T>& lhs, const std::pair<T, T>& rhs) {
    return std::make_pair(lhs.first - rhs.first, lhs.second - rhs.second);
}

// 日志追踪系统 - 现在在所有模式下都可用
class DebugTracker {
private:
    static std::map<std::string, int> callCount;
    static std::ofstream logFile;
    static bool enableLogging;
    static bool enableConsoleOutput;
    static int logLevel; // 0=关闭, 1=仅错误, 2=重要函数, 3=所有函数

public:
    // 设置日志级别
    static void setLogLevel(int level) {
        logLevel = level;
        enableLogging = (level > 0);
    }

    // 设置是否输出到控制台
    static void setConsoleOutput(bool enable) {
        enableConsoleOutput = enable;
    }

    // 初始化日志文件
    static bool initLogFile(const std::string& filename = "debug.log") {
        if (!enableLogging) return true;

        logFile.open(filename, std::ios::app); // 以追加模式打开
        if (!logFile.is_open()) {
            std::cerr << "[ERROR] Failed to open log file: " << filename << std::endl;
            return false;
        }
        log("Log file initialized: " + filename + " (Level: " + std::to_string(logLevel) + ")");
        return true;
    }

    // 关闭日志文件
    static void closeLogFile() {
        if (logFile.is_open()) {
            log("Closing log file");
            logFile.close();
        }
    }

    static void functionStart(const std::string& funcName, int requiredLevel = 3) {
        if (!enableLogging || logLevel < requiredLevel) return;

        callCount[funcName]++;
        log(funcName + " begin (call #" + std::to_string(callCount[funcName]) + ")");
    }

    static void functionEnd(const std::string& funcName, int requiredLevel = 3) {
        if (!enableLogging || logLevel < requiredLevel) return;
        log(funcName + " end");
    }

    static void log(const std::string& message, int requiredLevel = 1) {
        if (!enableLogging || logLevel < requiredLevel) return;

        // 获取当前时间
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        // 格式化时间
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
        std::string timestamp = ss.str();

        // 输出到控制台
        if (enableConsoleOutput) {
            std::cout << "[" << timestamp << " DEBUG] " << message << std::endl;
            std::cout.flush();
        }

        // 输出到文件（如果文件已打开）
        if (logFile.is_open()) {
            logFile << "[" << timestamp << " DEBUG] " << message << std::endl;
            logFile.flush();
        }
    }
};

// 静态成员初始化
std::map<std::string, int> DebugTracker::callCount;
std::ofstream DebugTracker::logFile;
bool DebugTracker::enableLogging = true;  // 默认开启，可通过环境变量控制
bool DebugTracker::enableConsoleOutput = false;  // 默认不输出到控制台
int DebugTracker::logLevel = 2;  // 默认级别2：重要函数

// 自动记录函数入口和退出的宏，支持不同级别
#define FUNC_ENTRY_CRITICAL() DebugTracker::functionStart(__func__, 1)  // 关键函数
#define FUNC_ENTRY_IMPORTANT() DebugTracker::functionStart(__func__, 2)  // 重要函数
#define FUNC_ENTRY() DebugTracker::functionStart(__func__, 3)  // 所有函数
#define FUNC_EXIT_CRITICAL() DebugTracker::functionEnd(__func__, 1)
#define FUNC_EXIT_IMPORTANT() DebugTracker::functionEnd(__func__, 2)
#define FUNC_EXIT() DebugTracker::functionEnd(__func__, 3)
#define LOG_CRITICAL(msg) DebugTracker::log(msg, 1)
#define LOG_IMPORTANT(msg) DebugTracker::log(msg, 2)
#define LOG(msg) DebugTracker::log(msg, 3)

#ifdef DEBUG
template <class T>
void debug(T _map[17][17]) {
    puts("");
    for(int i = 0; i < map_size; i++) {
        for(int j = 0; j < map_size; j++) {
            std::cout << _map[i][j] << " ";
        }
        std::cout << std::endl;
    }
    puts("");
}

template <class T>
void debug(vector <T> _map) {
    puts("");
    for(int i = 0; i < map_size; i++) {
        for(int j = 0; j < map_size; j++) {
            std::cout << _map[i][j] << " ";
        }
        std::cout << std::endl;
    }
    puts("");
}

template <class T>
void debug(T _map[289]) {
    puts("");
    for(int i = 0; i < 17; i++) {
        for(int j = 0; j < 17; j++) {
            std::cout << _map[i * map_size + j] << " ";
        }
        puts("");
    }

    std::cout << std::endl;
    puts("");
}
#endif

namespace FastSolve {
static constexpr int N = 17;
static constexpr int SIZE = N * N;          
static constexpr int R = 3;               
static constexpr int LMAX = 24;            
static constexpr int MOVES = 64;         
static constexpr int INVALID16 = 0xFFFF;

static int player_idx=-1, enemy_idx=-1;
static bool find_pos_status = -1;
static inline int idx_rc(int r, int c) { return r * N + c; }

static constexpr int8_t dR[4] = {-1, 0, 1, 0};
static constexpr int8_t dC[4] = { 0, 1, 0,-1};

static inline unsigned ctz64(uint64_t x) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    return (unsigned)_tzcnt_u64(x);
#elif defined(__BMI__) || defined(__BMI2__)
    return (unsigned)_tzcnt_u64(x);
#else
    return (unsigned)__builtin_ctzll(x);
#endif
}

static constexpr int MASK_WORDS = (SIZE + 63) / 64; // 5
alignas(64) uint64_t mask_region[SIZE][MASK_WORDS];
static bool g_mask_inited = false;

void init_region_masks() {
    FUNC_ENTRY();
    if (g_mask_inited) {
        FUNC_EXIT();
        return;
    }
    std::memset(mask_region, 0, sizeof(mask_region));
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            const int center = idx_rc(r, c);
            uint64_t* dst = mask_region[center];
            for (int dr = -R; dr <= R; ++dr) {
                int rr = r + dr;
                if ((unsigned)rr >= (unsigned)N) continue;
                int max_dc = R - (dr < 0 ? -dr : dr);
                for (int dc = -max_dc; dc <= max_dc; ++dc) {
                    int cc = c + dc;
                    if ((unsigned)cc >= (unsigned)N) continue;
                    int id = idx_rc(rr, cc);
                    dst[id >> 6] |= (uint64_t(1) << (id & 63));
                }
            }
        }
    }
    g_mask_inited = true;
    FUNC_EXIT();
}

static inline void find_positions_avx2(const int16_t* a) {
    FUNC_ENTRY();
    player_idx = -1;
    enemy_idx = -1;

    const __m256i player_v  = _mm256_set1_epi16((int16_t)-9);
    const __m256i enemy_v = _mm256_set1_epi16((int16_t)-2);

    int i = 0;
    for (; i + 15 < SIZE; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i player_eq  = _mm256_cmpeq_epi16(v, player_v);
        __m256i enemy_eq = _mm256_cmpeq_epi16(v, enemy_v);

        uint32_t player_mask  = (uint32_t)_mm256_movemask_epi8(player_eq);
        uint32_t enemy_mask = (uint32_t)_mm256_movemask_epi8(enemy_eq);

        if (player_idx < 0 && player_mask) {
            uint32_t mm = (player_mask & (player_mask >> 1)) & 0x55555555u;
#if defined(__BMI__) || defined(__BMI2__)
            unsigned bit = (unsigned)_tzcnt_u32(mm);
#else
            unsigned bit = (unsigned)__builtin_ctz(mm);
#endif
            player_idx = i + (int)(bit >> 1);
        }
        if (enemy_idx < 0 && enemy_mask) {
            uint32_t mm = (enemy_mask & (enemy_mask >> 1)) & 0x55555555u;
#if defined(__BMI__) || defined(__BMI2__)
            unsigned bit = (unsigned)_tzcnt_u32(mm);
#else
            unsigned bit = (unsigned)__builtin_ctz(mm);
#endif
            enemy_idx = i + (int)(bit >> 1);
        }
        if (player_idx >= 0 && enemy_idx >= 0) {
            find_pos_status = true;
            LOG("Found both players at positions: player=" + std::to_string(player_idx) + ", enemy=" + std::to_string(enemy_idx));
            FUNC_EXIT();
            return;
        }
    }
    for (; i < SIZE; ++i) {
        int16_t v = a[i];
        if (player_idx < 0 && v == -9) player_idx = i;
        if (enemy_idx < 0 && v == -2) enemy_idx = i;
        if (player_idx >= 0 && enemy_idx >= 0) {
            find_pos_status = true;
            LOG("Found both players at positions: player=" + std::to_string(player_idx) + ", enemy=" + std::to_string(enemy_idx));
            FUNC_EXIT();
            return;
        }
    }
    find_pos_status = false;
    LOG("Failed to find both players");
    FUNC_EXIT();
}


bool overlap_flag = false;
static inline void overlap_positive_sum_gt10(const int16_t grid[N][N]) {
    FUNC_ENTRY();
    const int16_t* a = &grid[0][0];
    if (!find_pos_status) {
        overlap_flag = false;
        LOG("find_pos_status is false, setting overlap_flag to false");
        FUNC_EXIT();
        return;
    }

    const uint64_t* m1 = mask_region[player_idx];
    const uint64_t* m2 = mask_region[enemy_idx];

    uint64_t overlap[MASK_WORDS];
#pragma unroll
    for (int w = 0; w < MASK_WORDS; ++w) {
        overlap[w] = m1[w] & m2[w];
    }

    if ((overlap[0] | overlap[1] | overlap[2] | overlap[3] | overlap[4]) == 0ull) {
        overlap_flag = false;
        LOG("No overlap found between player and enemy regions");
        FUNC_EXIT();
        return;
    }

    const int16_t* aBase = a;
    int sum = 0;
#pragma unroll
    for (int w = 0; w < MASK_WORDS; ++w) {
        uint64_t x = overlap[w];
        while (x) {
            unsigned b = ctz64(x);
            x &= (x - 1);
            int idx = (w << 6) + (int)b;
            if ((unsigned)idx >= (unsigned)SIZE) break;
            int16_t v = aBase[idx];
            if (v > 0) {
                sum += (int)v;
                if (sum > 7) overlap_flag = true;
            }
        }
    }
    // printf("sum=%d\n", sum);
    overlap_flag =  sum > 7;
    LOG("Overlap sum: " + std::to_string(sum) + ", overlap_flag: " + (overlap_flag ? "true" : "false"));
    FUNC_EXIT();
}

static inline pair<int,int> max64_avx2(const int* a) {
    FUNC_ENTRY();
    const __m256i idx_base = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
    __m256i maxv = _mm256_load_si256((const __m256i*)(a));
    __m256i max_idx = idx_base;
    for (int base = 8; base < 64; base += 8) {
        __m256i v = _mm256_load_si256((const __m256i*)(a + base));
        __m256i new_idx = _mm256_add_epi32(idx_base, _mm256_set1_epi32(base));
        __m256i gt = _mm256_cmpgt_epi32(v, maxv);
        maxv     = _mm256_blendv_epi8(maxv, v, gt);
        max_idx  = _mm256_blendv_epi8(max_idx, new_idx, gt);
    }
    alignas(64) int vals[8], idxs[8];
    _mm256_store_si256((__m256i*)vals, maxv);
    _mm256_store_si256((__m256i*)idxs, max_idx);
    int best_val = vals[0], best_idx = idxs[0];
    for (int i = 1; i < 8; ++i) {
        if (vals[i] > best_val) {
            best_val = vals[i];
            best_idx = idxs[i];
        }
    }
    LOG("max64_avx2 found best_val=" + std::to_string(best_val) + ", best_idx=" + std::to_string(best_idx));
    FUNC_EXIT();
    return {best_val, best_idx};
}

alignas(64) static uint16_t local_to_global[SIZE][LMAX];

alignas(64) static uint8_t u0_tab[SIZE][MOVES];
alignas(64) static uint8_t u1_tab[SIZE][MOVES];
alignas(64) static uint8_t u2_tab[SIZE][MOVES];
alignas(64) static uint8_t k_tab [SIZE][MOVES];

alignas(64) static uint8_t valid_static[SIZE][MOVES];

alignas(64) static uint32_t subset_mask24[SIZE][MOVES];

static bool g_paths_inited = false;

static void build_local_map_for_start(int sr, int sc, uint16_t out_local_to_global[LMAX]) {
    // Note: This function is called very frequently during initialization, using minimal logging
    for (int i = 0; i < LMAX; ++i) out_local_to_global[i] = INVALID16;

    int cnt = 0;
    for (int d = 1; d <= 3; ++d) {
        for (int dr = -d; dr <= d; ++dr) {
            int dc = d - (dr < 0 ? -dr : dr);
            for (int sgn = 0; sgn < 2; ++sgn) {
                int rr = sr + dr;
                int cc = sc + (sgn == 0 ? dc : -dc);
                if (dc == 0 && sgn == 1) continue;
                if ((unsigned)rr < (unsigned)N && (unsigned)cc < (unsigned)N) {
                    if (cnt < LMAX)
                        out_local_to_global[cnt++] = (uint16_t)idx_rc(rr, cc);
                }
            }
        }
    }
}

struct LocalIndexMap {
    uint8_t g2l[SIZE]; 
    inline void reset() {
        std::memset(g2l, 0xFF, sizeof(g2l));
    }

    inline void fill_from_local_to_global(const uint16_t ltg[LMAX]) {
#pragma unroll
        for (int i = 0; i < LMAX; ++i) {
            uint16_t g = ltg[i];
            if (g != INVALID16) g2l[g] = (uint8_t)i;
        }
    }
};

void init_paths_first_round(const int16_t grid[N][N]) {
    FUNC_ENTRY();
    init_region_masks();

    LocalIndexMap lmap;

    for (int sr = 0; sr < N; ++sr) {
        for (int sc = 0; sc < N; ++sc) {
            const int sidx = idx_rc(sr, sc);
            build_local_map_for_start(sr, sc, local_to_global[sidx]);

            lmap.reset();
            lmap.fill_from_local_to_global(local_to_global[sidx]);

            for (int m = 0; m < MOVES; ++m) {
                int dir0 = (m     ) & 3;
                int dir1 = (m >> 2) & 3;
                int dir2 = (m >> 4) & 3;

                int r = sr, c = sc;
                bool ok = true;

                int r1 = r + dR[dir0], c1 = c + dC[dir0];
                if ((unsigned)r1 >= (unsigned)N || (unsigned)c1 >= (unsigned)N) ok = false;
                else if (grid[r1][c1] == -1) ok = false;

                int r2 = 0, c2 = 0, r3 = 0, c3 = 0;
                if (ok) {
                    r2 = r1 + dR[dir1]; c2 = c1 + dC[dir1];
                    if ((unsigned)r2 >= (unsigned)N || (unsigned)c2 >= (unsigned)N) ok = false;
                    else if (grid[r2][c2] == -1) ok = false;
                }
                if (ok) {
                    r3 = r2 + dR[dir2]; c3 = c2 + dC[dir2];
                    if ((unsigned)r3 >= (unsigned)N || (unsigned)c3 >= (unsigned)N) ok = false;
                    else if (grid[r3][c3] == -1) ok = false;
                }

                if (!ok) {
                    valid_static[sidx][m] = 0;
                    u0_tab[sidx][m] = u1_tab[sidx][m] = u2_tab[sidx][m] = 0;
                    k_tab [sidx][m] = 0;
                    subset_mask24[sidx][m] = 0;
                    continue;
                }

                const int idx1 = idx_rc(r1, c1);
                const int idx2 = idx_rc(r2, c2);
                const int idx3 = idx_rc(r3, c3);

                uint8_t l1 = lmap.g2l[idx1];
                uint8_t l2 = lmap.g2l[idx2];
                uint8_t l3 = lmap.g2l[idx3];

                uint8_t U[3];
                int k = 0;

                U[k++] = l1;
                if (l2 != l1) U[k++] = l2;
                if (l3 != l1 && l3 != l2) U[k++] = l3;

                u0_tab[sidx][m] = U[0];
                u1_tab[sidx][m] = (k >= 2) ? U[1] : U[0];
                u2_tab[sidx][m] = (k == 3) ? U[2] : ((k == 2) ? U[1] : U[0]);
                k_tab [sidx][m] = (uint8_t)k;
                valid_static[sidx][m] = 1;

                uint32_t mask = (1u << U[0]);
                if (k >= 2) mask |= (1u << U[1]);
                if (k == 3) mask |= (1u << U[2]);
                subset_mask24[sidx][m] = mask;
            }
        }
    }

    g_paths_inited = true;
    LOG("Paths initialization completed");
    FUNC_EXIT();
}

static inline pair<int,int> best3step_move_fast(const int16_t grid[N][N]) {
    FUNC_ENTRY();
    alignas(64) uint16_t pos_vals_u16[LMAX];
    uint32_t obst3_mask = 0;

    const uint16_t* ltg = local_to_global[player_idx];

    for (int i = 0; i < LMAX; ++i) {
        uint16_t g = ltg[i];
        if (g == INVALID16) {
            pos_vals_u16[i] = 0;
            continue;
        }
        int16_t v = ((const int16_t*)&grid[0][0])[g];
        if (v == -3 || v == -2) obst3_mask |= (1u << i);
        pos_vals_u16[i] = (uint16_t)(v > 0 ? v : 0);
    }

    alignas(64) int scores[64];
    const uint8_t* __restrict u0 = u0_tab[player_idx];
    const uint8_t* __restrict u1 = u1_tab[player_idx];
    const uint8_t* __restrict u2 = u2_tab[player_idx];
    const uint8_t* __restrict kk = k_tab [player_idx];
    const uint8_t* __restrict vs = valid_static[player_idx];
    const uint32_t* __restrict sm = subset_mask24[player_idx];

#pragma unroll
    for (int m = 0; m < MOVES; m += 4) {
    if (!vs[m] || (sm[m] & obst3_mask)) {
        scores[m] = std::numeric_limits<int>::min();
    } else {
        uint32_t s = pos_vals_u16[u0[m]];
        if (kk[m] >= 2) s += pos_vals_u16[u1[m]];
        if (kk[m] >= 3) s += pos_vals_u16[u2[m]];
        scores[m] = (int)s;
    }

    if (!vs[m+1] || (sm[m+1] & obst3_mask)) {
        scores[m+1] = std::numeric_limits<int>::min();
    } else {
        uint32_t s = pos_vals_u16[u0[m+1]];
        if (kk[m+1] >= 2) s += pos_vals_u16[u1[m+1]];
        if (kk[m+1] >= 3) s += pos_vals_u16[u2[m+1]];
        scores[m+1] = (int)s;
    }

    if (!vs[m+2] || (sm[m+2] & obst3_mask)) {
        scores[m+2] = std::numeric_limits<int>::min();
    } else {
        uint32_t s = pos_vals_u16[u0[m+2]];
        if (kk[m+2] >= 2) s += pos_vals_u16[u1[m+2]];
        if (kk[m+2] >= 3) s += pos_vals_u16[u2[m+2]];
        scores[m+2] = (int)s;
    }
    
    if (!vs[m+3] || (sm[m+3] & obst3_mask)) {
        scores[m+3] = std::numeric_limits<int>::min();
    } else {
        uint32_t s = pos_vals_u16[u0[m+3]];
        if (kk[m+3] >= 2) s += pos_vals_u16[u1[m+3]];
        if (kk[m+3] >= 3) s += pos_vals_u16[u2[m+3]];
        scores[m+3] = (int)s;
    }
}

    auto result = max64_avx2(scores);
    LOG("best3step_move_fast result: value=" + std::to_string(result.first) + ", move=" + std::to_string(result.second));
    FUNC_EXIT();
    return result;
}

pair<int,int> best3step_entry(const int16_t grid[N][N]) {
    FUNC_ENTRY();
    if (!g_paths_inited) {
        LOG("Paths not initialized, calling init_paths_first_round");
        init_paths_first_round(grid);
    }

    const int16_t* a = &grid[0][0];
    if (!find_pos_status) {
        LOG("find_pos_status is false, returning min value");
        FUNC_EXIT();
        return {std::numeric_limits<int>::min(), -1};
    }

    if (!overlap_flag) {
        LOG("overlap_flag is false, returning min value");
        FUNC_EXIT();
        return {std::numeric_limits<int>::min(), -1};
    }

    auto result = best3step_move_fast(grid);
    LOG("best3step_entry result: value=" + std::to_string(result.first) + ", move=" + std::to_string(result.second));
    FUNC_EXIT();
    return result;
}
inline __attribute__((always_inline))
void convert_single_row_avx2(const int* __restrict__ in_row, int16_t* __restrict__ out_row) {
    // Note: This function is called very frequently, so we use minimal logging
    __m256i v32_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_row));
    __m128i v16_1 = _mm_packs_epi32(_mm256_castsi256_si128(v32_1), _mm256_extracti128_si256(v32_1, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_row), v16_1);

    __m256i v32_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_row + 8));
    __m128i v16_2 = _mm_packs_epi32(_mm256_castsi256_si128(v32_2), _mm256_extracti128_si256(v32_2, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_row + 8), v16_2);

    out_row[16] = static_cast<int16_t>(in_row[16]);
}


inline __attribute__((always_inline))
void cvt2int16(const std::vector<std::vector<int>>& grid, int16_t res[N][N]) {
    FUNC_ENTRY();
    _mm_prefetch(grid[1].data(), _MM_HINT_T0);
    _mm_prefetch(grid[9].data(), _MM_HINT_T0);
    convert_single_row_avx2(grid[0].data(), res[0]);
    convert_single_row_avx2(grid[8].data(), res[8]);

    _mm_prefetch(grid[2].data(), _MM_HINT_T0);
    _mm_prefetch(grid[10].data(), _MM_HINT_T0);
    convert_single_row_avx2(grid[1].data(), res[1]);
    convert_single_row_avx2(grid[9].data(), res[9]);

    _mm_prefetch(grid[3].data(), _MM_HINT_T0);
    _mm_prefetch(grid[11].data(), _MM_HINT_T0);
    convert_single_row_avx2(grid[2].data(), res[2]);
    convert_single_row_avx2(grid[10].data(), res[10]);

    _mm_prefetch(grid[4].data(), _MM_HINT_T0);
    _mm_prefetch(grid[12].data(), _MM_HINT_T0);
    convert_single_row_avx2(grid[3].data(), res[3]);
    convert_single_row_avx2(grid[11].data(), res[11]);

    _mm_prefetch(grid[5].data(), _MM_HINT_T0);
    _mm_prefetch(grid[13].data(), _MM_HINT_T0);
    convert_single_row_avx2(grid[4].data(), res[4]);
    convert_single_row_avx2(grid[12].data(), res[12]);

    _mm_prefetch(grid[6].data(), _MM_HINT_T0);
    _mm_prefetch(grid[14].data(), _MM_HINT_T0);
    convert_single_row_avx2(grid[5].data(), res[5]);
    convert_single_row_avx2(grid[13].data(), res[13]);

    _mm_prefetch(grid[7].data(), _MM_HINT_T0);
    _mm_prefetch(grid[15].data(), _MM_HINT_T0);
    _mm_prefetch(grid[16].data(), _MM_HINT_T0);
    convert_single_row_avx2(grid[6].data(), res[6]);
    convert_single_row_avx2(grid[14].data(), res[14]);

    convert_single_row_avx2(grid[7].data(), res[7]);
    convert_single_row_avx2(grid[15].data(), res[15]);

    convert_single_row_avx2(grid[16].data(), res[16]);
    LOG("Grid conversion to int16 completed");
    FUNC_EXIT();
}
// return value ,opcode

alignas(64) static int16_t res[N][N];
inline void cvt2int16(const std::vector<std::vector<int>>& grid) {
    cvt2int16(grid, res);
}

inline pair<int,int> best3step_entry() {
    return best3step_entry(res);
}

}


const int INF = 0x3f3f3f3f;


vector <Position> available_positions;


void print_reward() {
    FUNC_ENTRY();
    for(int i = 0; i < map_size; i++) {
        for(int j = 0; j < map_size; j++) {
            std::cout << reward[i][j] << " ";
        }
        std::cout << std::endl;
    }
    LOG("Reward matrix printed");
    FUNC_EXIT();
}


inline int get_next_npc(int round) {
    // need to further implement this function
    if(!npc_appear_round) return -1;
    int res = ((npc_appear_round + 1) - (round % npc_appear_round)) % npc_appear_round;
    if(res == npc_appear_round) res = 0;
    return res;
}


inline bool in9x9(const Position& pos){
    return pos.first >= 4 && pos.first <= 12 && pos.second >= 4 && pos.second <= 12;
}

inline bool in5x5(const Position& pos){
    return pos.first >= 6 && pos.first <= 10 && pos.second >= 6 && pos.second <= 10;
}

inline void flag_npc_area(const Map& _map) {
    FUNC_ENTRY();
    memset(npc_area, 0, sizeof(npc_area));
    int npc_count = 0;
    for(int i = 0; i < map_size; i++) {
        for(int j = 0; j < map_size; j++) {
           if(!in9x9({i, j}) && _map[i][j] > 0) {
               npc_area[i][j] = true;
               npc_count++;
           }
       }
   }
   LOG("Flagged " + std::to_string(npc_count) + " NPC area positions");
   FUNC_EXIT();
}

inline bool in_npc_area(Position pos) {
    return npc_area[pos.first][pos.second];
}


// Now this is just a simple example that  need to be strengthened
inline double get_extra_reward(const Position &pos, const double cur_reward, int round) {
    // Note that the center 5*5 may higher than 9*9
    // And there are some girds that explosive growth in few rounds
    // For NPC, considering the round it occurs.
    
    if(cur_reward == 0.0) return 0.0;
    double extra_reward = 0.0;
    static double center9_reward = 0.5;
    static double center5_reward = 0.3;
    

    extra_reward += center9_reward * in9x9(pos);
    extra_reward += center5_reward * in5x5(pos);
    int need = get_next_npc(round);
    if(need <= 1) extra_reward += 0.4;
    else {
        extra_reward += in_npc_area(pos) *  0.4 / need;
    }
    return extra_reward;
}

inline bool check_position(const Position& pos, const int _map[17][17]) {
    return pos.first < 0 || pos.first >= map_size || pos.second < 0 || pos.second >= map_size || _map[pos.first][pos.second] < 0;
}

int absolute_reward[17][17];


enum Status {
    BOOM_CHANGED = 1
};

#define IGNORE_LIMIT 59

bool vis[17][17];
inline void reward_block(int _map[17][17], int _round) {
    FUNC_ENTRY();
    memset(vis, 0, sizeof(vis));
    int sum;
    std::vector <Position> block;
    std::queue <Position> q;
    int npc_rest_round = get_next_npc(_round);
    // npc_buff / 5, normal is 5, and 6 means 15% buff of npc.
    int npc_buff = 0;
    if(npc_rest_round < 5 && npc_rest_round > -1) {
        npc_buff = (5 - npc_rest_round);
    }
    LOG("Reward block processing: round=" + std::to_string(_round) + ", npc_rest_round=" + std::to_string(npc_rest_round) + ", npc_buff=" + std::to_string(npc_buff));

    for(int i = 0; i < map_size; i++) {
        for(int j = 0; j < map_size; j++) {
            if(_map[i][j] <= 0) continue;
            if(vis[i][j]) continue;
            // puts("------------------");
            // printf("i = %d, j = %d\n", i, j);
            block.clear();
            sum = 0;
            q.push({i, j});
            vis[i][j] = true;
            while(q.size()) {
                auto u = q.front();
                q.pop();
                
                sum += _map[u.first][u.second];
                block.push_back(u);
                for(int k = 0; k < 4; k++) {
                    Position v = {u.first + directions[k].first, u.second + directions[k].second};
                    if((unsigned)v.first >= map_size || (unsigned)v.second >= map_size) continue;
                    if(_map[v.first][v.second] <= 0) continue;
                    if(vis[v.first][v.second]) continue;
                    vis[v.first][v.second] = true;
                    q.push(v);
                }
            }

            int cnt = block.size();
            int ave = sum / cnt;
           // printf("sum =%d, cnt = %d, ave = %d\n", sum, cnt, ave);
            if(sum < 10 || ave < 4) {
                continue;
            }
            int rwd = (sum - 3 * cnt) / cnt / 2;
            rwd += (ave >> 1);
           // printf("rwd = %d\n", rwd);
            for(auto &it : block) {
                _map[it.first][it.second] += rwd;
            }
        }
    }
    for(int i = 0; i < map_size; i++) {
        for(int j = 0; j < map_size; j++) {
           if(in_npc_area({i, j}) && _map[i][j] > 0) {
               _map[i][j] += (npc_buff * _map[i][j]) * 3 / 20;
           }
        }
    }
    LOG("Reward block processing completed");
    FUNC_EXIT();
}

inline int huge_reward(const int &r) {
    if(r >= 14) {
        return r + (r - 14);
    }
    return r;
}

inline void bfs(const Position& src, int dis[289], int _map[17][17]) {
    FUNC_ENTRY();
    for(int i = 0; i < 289; i++) {
        dis[i] = INF;
    }
    dis[src.first * map_size + src.second] = 0;
    queue <int> q;
    q.push(src.first * map_size + src.second);
    static bool inq[289];
    memset(inq, 0, sizeof(inq));
    inq[src.first * map_size + src.second] = true;
    while(!q.empty()) {
        int curr = q.front();
        q.pop();
        int x = curr / map_size;
        int y = curr % map_size;
        for(int i = 0; i < 4; i++) {
            Position new_pos = {x + directions[i].first, y + directions[i].second};
            if(check_position(new_pos, _map)) continue;
            int new_idx = new_pos.first * map_size + new_pos.second;
            if(inq[new_idx]) continue;
            if(dis[new_idx] > dis[curr] + 1) {
                dis[new_idx] = dis[curr] + 1;
                q.push(new_idx);
                inq[new_idx] = true;
            }
        }
    }
    LOG("BFS completed from position (" + std::to_string(src.first) + "," + std::to_string(src.second) + ")");
    FUNC_EXIT();
}

inline void init_map(const Map& _map, int new_map[17][17], const int& coin, int _round) {
    FUNC_ENTRY();
    bool ignore = coin < IGNORE_LIMIT;
    LOG("Initializing map with coin=" + std::to_string(coin) + ", round=" + std::to_string(_round) + ", ignore=" + (ignore ? "true" : "false"));

    for(int i = 0; i < map_size; i++) {
        for(int j = 0; j < map_size; j++) {
            if(_map[i][j] == -9) {
                new_map[i][j] = 0;
            } else if(_map[i][j] == -3) {
                new_map[i][j] = ignore ? 0 : -300;
            } else if(_map[i][j] > 0 && in5x5({i, j})) {
                new_map[i][j] = (_map[i][j] * 5) >> 2;
            } else if(_map[i][j] > 0 && in9x9({i, j})) {
                new_map[i][j] = (_map[i][j] * 5) >> 2;
            } else {
                new_map[i][j] = _map[i][j];
            }
            new_map[i][j] = huge_reward(new_map[i][j]);
        }
    }
   reward_block(new_map, _round);
   static int dis_player[289], dis_enemy[289];
   Position player_pos = {FastSolve::player_idx / map_size, FastSolve::player_idx % map_size};
   Position enemy_pos = {FastSolve::enemy_idx / map_size, FastSolve::enemy_idx % map_size};
   LOG("Player position: (" + std::to_string(player_pos.first) + "," + std::to_string(player_pos.second) + ")");
   LOG("Enemy position: (" + std::to_string(enemy_pos.first) + "," + std::to_string(enemy_pos.second) + ")");
   bfs(player_pos, dis_player, new_map);
   bfs(enemy_pos, dis_enemy, new_map);
   for(int i = 0; i < map_size; i++) {
       for(int j = 0; j < map_size; j++) {
           int idx = i * map_size + j;
           if(dis_player[idx] != INF && dis_enemy[idx] != INF) {
                if(dis_enemy[idx] <= 3 && dis_player[idx] > 3) {
                    new_map[i][j] = 1;
                }
           }
       }
   }
   FUNC_EXIT();
}

double explore_reward[17][17];
const double enemy_advance_decay = 0.25;
const double decay_factor = 0.8;

int check_status(const Map& old_map, const Map& new_map) {
    FUNC_ENTRY();
    static int status = 0;
    for(int i = 0; i < map_size; i++) {
        for(int j = 0; j < map_size; j++) {
            bool is_boom1 = (old_map[i][j] == -3);
            bool is_boom2 = (new_map[i][j] == -3);
            if(is_boom1 ^ is_boom2) {
                status |= BOOM_CHANGED;
                LOG("Boom status changed at position (" + std::to_string(i) + "," + std::to_string(j) + ")");
            }
            //other check need to be implemented
        }
    }
    LOG("Status check result: " + std::to_string(status));
    FUNC_EXIT();
    return status;
}
void check_npc(const Map& old_map, const Map& new_map, const int& round_now) {
    FUNC_ENTRY();
    for(int i = 0; i < map_size; i++) {
        for(int j = 0; j < map_size; j++) {
            if(npc_area[i][j] && new_map[i][j] > old_map[i][j] && old_map[i][j] >= 0 && new_map[i][j] >= 0) {
                npc_appear_round = round_now - 1;
                LOG("NPC detected at position (" + std::to_string(i) + "," + std::to_string(j) + "), npc_appear_round set to " + std::to_string(npc_appear_round));
                FUNC_EXIT();
                return;
            }
        }
    }
    LOG("No NPC activity detected");
    FUNC_EXIT();
}

inline Position do_onemove(const Position& pos, int op) {
    return {pos.first + directions[op].first, pos.second + directions[op].second};
}

double pw_decay[32];

bool stop_explore;
inline double get_explore_score(int _map[17][17], const Position& src, const double& cur_reward, const int& _round) {
    FUNC_ENTRY();
    //if(stop_explore) return 0.0;
    // if(cur_reward >= 20.0) {
    //     stop_explore = true;
    //     return cur_reward;
    // }
    double res = 0.0;
    static int dis[289];
    bfs(src, dis, _map);
    for(int i = 0; i < map_size; i++) {
        for(int j = 0; j < map_size; j++) {
            if(_map[i][j] < 0) continue;
            if(dis[i * map_size + j] == INF) continue; // unreachable
            const double factor = dis[i * map_size + j] > 31 ? pw_decay[31] : pw_decay[dis[i * map_size + j]];
            res += 1.0 * _map[i][j] * factor;
        }
    }
    double scale = 5.0;
    // if(cur_reward >= 0 && cur_reward < 3) {
    //     scale += (5 - cur_reward);
    // }
    res /= scale;
    if(_round > 293) {
        res /= 3.0;
    }
    LOG("Explore score for position (" + std::to_string(src.first) + "," + std::to_string(src.second) + "): " + std::to_string(res));
    FUNC_EXIT();
    return res;
}


inline double total_reward(const Position& pos, int _map[17][17], const double& cur_reward, int _round) {
    FUNC_ENTRY();
    auto bak = _map[pos.first][pos.second];
    _map[pos.first][pos.second] = 0;
    auto expand = get_explore_score(_map, pos, cur_reward, _round);
    double res = cur_reward + bak + expand;

    _map[pos.first][pos.second] = bak;
    LOG("Total reward for position (" + std::to_string(pos.first) + "," + std::to_string(pos.second) + "): cur=" + std::to_string(cur_reward) + ", bak=" + std::to_string(bak) + ", expand=" + std::to_string(expand) + ", total=" + std::to_string(res));
    FUNC_EXIT();
    return res;
}


#define CHECKPASS !check_position(pos3, _map)


void get_best_move(const Position& pos, const Position& enemy, int _map[17][17], int *res, int _round) {
    FUNC_ENTRY();
    LOG("Getting best move for position (" + std::to_string(pos.first) + "," + std::to_string(pos.second) + "), round=" + std::to_string(_round));

    // stop_explore = false;
    res[0] = 4, res[1] = 4, res[2] = 4;

    double best_reward = 0.0;
    double cur_reward;

    for(int step1 = 0; step1 < 4; step1++) {
        auto pos1 = do_onemove(pos, step1);
        if(check_position(pos1, _map)) continue;
        cur_reward = _map[pos1.first][pos1.second];
        auto absolute_reward_bak1 = _map[pos1.first][pos1.second];
        _map[pos1.first][pos1.second] = 0.0;
        const double cur_reward_lst = cur_reward;
        for(int step2 = 0; step2 < 4; step2++) {
            auto pos2 = do_onemove(pos1, step2);
            if(check_position(pos2, _map)) continue;
            cur_reward = cur_reward_lst + _map[pos2.first][pos2.second];
            auto absolute_reward_bak2 = _map[pos2.first][pos2.second];
            _map[pos2.first][pos2.second] = 0.0;
            //We can use step3 as a for loop, but we unroll it for better performance

            auto pos3 = do_onemove(pos2, 0);
           
            if(CHECKPASS) {
               // printf("%d %d 0:", step1, step2);
                auto tot = total_reward(pos3, _map, cur_reward, _round);
                if(tot > best_reward) {
                    best_reward = tot;
                    res[0] = step1;
                    res[1] = step2;
                    res[2] = 0;
                }
            }
            

            pos3 = do_onemove(pos2, 1);
            if(CHECKPASS) {
             //   printf("%d %d 1:", step1, step2);
                auto tot = total_reward(pos3, _map, cur_reward, _round);
                if(tot > best_reward) {
                    best_reward = tot;
                    res[0] = step1;
                    res[1] = step2;
                    res[2] = 1;
                }
            }
            

            pos3 = do_onemove(pos2, 2);
            if(CHECKPASS) {
             //   printf("%d %d 2:", step1, step2);
                auto tot = total_reward(pos3, _map, cur_reward, _round);
                if(tot > best_reward) {
                    best_reward = tot;
                    res[0] = step1;
                    res[1] = step2;
                    res[2] = 2;
                }
            }

            pos3 = do_onemove(pos2, 3);
            if(CHECKPASS) {
               // printf("%d %d 3:", step1, step2);
                auto tot = total_reward(pos3, _map, cur_reward, _round);
                if(tot > best_reward) {
                    best_reward = tot;
                    res[0] = step1;
                    res[1] = step2;
                    res[2] = 3;
                }
            }

            _map[pos2.first][pos2.second] = absolute_reward_bak2;
        }

        _map[pos1.first][pos1.second] = absolute_reward_bak1;
    }

    LOG("Best move found: [" + std::to_string(res[0]) + "," + std::to_string(res[1]) + "," + std::to_string(res[2]) + "] with reward=" + std::to_string(best_reward));
    FUNC_EXIT();
}


void find_player(const Map& _map, Position &player, Position &enemy) {
    FUNC_ENTRY();
    for (int i = 0; i < map_size; ++i) {
        for (int j = 0; j < map_size; ++j) {
            if (_map[i][j] == -9) {
                player = {i, j};
            } else if (_map[i][j] == -2) {
                enemy = {i, j};
            }
        }
    }
    LOG("Found player at (" + std::to_string(player.first) + "," + std::to_string(player.second) + "), enemy at (" + std::to_string(enemy.first) + "," + std::to_string(enemy.second) + ")");
    FUNC_EXIT();
}

bool have_run = false;

Map prev_map;


int last_coin, zero_cnt;

int round_now = 0;

int new_grid[17][17];

#include <immintrin.h>
#include <utility>   
#include <cstdint>   

// 假设的类型和常量
using Position = std::pair<int, int>;
using Direction = std::pair<int, int>;


inline void greedy(const Position& pos, const int _map[17][17], Position* pred) {
    FUNC_ENTRY();
    int max_val = -INF;
    for(int i = 0; i < 4; i++) {
        auto pos1 = pos + directions[i];
        if(check_position(pos1, _map)) continue;
        for(int j = 0; j < 4; j++) {
            auto pos2 = pos1 + directions[j];
            if(check_position(pos2, _map)) continue;
            for(int k = 0; k < 4; k++) {
                auto pos3 = pos2 + directions[k];
                if(check_position(pos3, _map)) continue;
                int val = _map[pos1.first][pos1.second];
                if(pos2 != pos1) {
                    val += _map[pos2.first][pos2.second];
                }
                if(pos3 != pos2 && pos3 != pos1) {
                    val += _map[pos3.first][pos3.second];
                }

                if(val > max_val) {
                    max_val = val;
                    pred[0] = pos1;
                    pred[1] = pos2;
                    pred[2] = pos3;
                }
            }
        }
    }
    LOG("Greedy search from (" + std::to_string(pos.first) + "," + std::to_string(pos.second) + ") found max_val=" + std::to_string(max_val));
    FUNC_EXIT();
}

alignas(32) int nxt_op[8];
int predicted = 0;
int pred_res[3];
inline void get_next_move(const Position& pos, const Position& enemy, int _map[17][17], \
    const int *step, int _round) {
    FUNC_ENTRY();
    LOG("Getting next move for position (" + std::to_string(pos.first) + "," + std::to_string(pos.second) + "), steps=[" + std::to_string(step[0]) + "," + std::to_string(step[1]) + "," + std::to_string(step[2]) + "]");

    int new_map[17][17];
    int res[3] = {4, 4, 4};
    for(int i = 0; i < map_size; i++) {
        for(int j = 0; j < map_size; j++) {
            new_map[i][j] = _map[i][j];
        }
    }
    _map[enemy.first][enemy.second] = 0;
    static Position pred[3];
    greedy(enemy, _map, pred);
    auto pos1 = pos + directions[step[0]];
    new_map[pos1.first][pos1.second] = 0;
    auto pos2 = pos1 + directions[step[1]];
    new_map[pos2.first][pos2.second] = 0;
    auto pos3 = pos2 + directions[step[2]];
    new_map[pos3.first][pos3.second] = 0;

    new_map[pred[0].first][pred[0].second] = 0;
    new_map[pred[1].first][pred[1].second] = 0;
    new_map[pred[2].first][pred[2].second] = -2;

    get_best_move(pos3, pred[2], new_map, res, _round);
    nxt_op[0] = directions[res[0]].first;
    nxt_op[1] = directions[res[0]].second;
    nxt_op[2] = directions[res[1]].first;
    nxt_op[3] = directions[res[1]].second;
    nxt_op[4] = directions[res[2]].first;
    nxt_op[5] = directions[res[2]].second;
    nxt_op[6] = 0;
    nxt_op[7] = 0;
    pred_res[0] = res[0], pred_res[1] = res[1], pred_res[2] = res[2];
    
    pos1 = pos3 + directions[res[0]];
    pos2 = pos1 + directions[res[1]];
    pos3 = pos2 + directions[res[2]];
    predicted = _map[pos1.first][pos1.second];
    if(pos2 != pos1) {
        predicted += _map[pos2.first][pos2.second];
    }
    if(pos3 != pos2 && pos3 != pos1) {
        predicted += _map[pos3.first][pos3.second];
    }
    if(predicted < 0) predicted = 0;
    LOG("Next move prediction: [" + std::to_string(pred_res[0]) + "," + std::to_string(pred_res[1]) + "," + std::to_string(pred_res[2]) + "], predicted reward=" + std::to_string(predicted));
    FUNC_EXIT();
}


inline bool fast_check(const Position& pos, const int16_t _map[17][17]) {
    FUNC_ENTRY();
    if(!predicted) {
        LOG("No prediction available, returning false");
        FUNC_EXIT();
        return false;
    }
    int tmp = predicted;
    predicted = 0;
    LOG("Fast check for position (" + std::to_string(pos.first) + "," + std::to_string(pos.second) + "), predicted=" + std::to_string(tmp));
    __m128i pos_xy = _mm_set_epi32(0, 0, pos.second, pos.first);
    __m256i v_pos = _mm256_broadcastsi128_si256(pos_xy);

    __m256i v_op = _mm256_loadu_si256((__m256i*)nxt_op);

    __m256i v_coords = _mm256_add_epi32(v_pos, v_op);
    const __m256i v_map_size_minus_1 = _mm256_set1_epi32(map_size - 1);
    __m256i mask_ge = _mm256_cmpgt_epi32(v_coords, v_map_size_minus_1);
    const __m256i v_zero = _mm256_setzero_si256();
    __m256i mask_lt = _mm256_cmpgt_epi32(v_zero, v_coords);

    __m256i oob_mask = _mm256_or_si256(mask_ge, mask_lt);

    if (!_mm256_testz_si256(oob_mask, oob_mask)) {
        return false;
    }

    alignas(32) int coords[8];
    _mm256_store_si256((__m256i*)coords, v_coords);
    
    const int x1 = coords[0], y1 = coords[1];
    const int x2 = coords[2], y2 = coords[3];
    const int x3 = coords[4], y3 = coords[5];

    int check_sum = _map[x1][y1];
    if (x1 != x2 || y1 != y2) {
        check_sum += _map[x2][y2];
    }
    if ((x3 != x2 || y3 != y2) && (x3 != x1 || y3 != y1)) {
        check_sum += _map[x3][y3];
    }

    if(_map[x1][y1] == -3 || _map[x2][y2] == -3 || _map[x3][y3] == -3) [[unlikely]] {
        LOG("Found obstacle (-3) in path, returning false");
        FUNC_EXIT();
        return false;
    }

    bool result = check_sum >= tmp || check_sum > 5;
    LOG("Fast check result: check_sum=" + std::to_string(check_sum) + ", tmp=" + std::to_string(tmp) + ", result=" + (result ? "true" : "false"));
    FUNC_EXIT();
    return result;
}

inline int reflect(const int &x) {
    return ((x & 1) << 1) | ((x >> 1) ^ (x & 1));
}

class Player
{
public:
    Player() {
        FUNC_ENTRY();
        srand(time(0));
        LOG("Player instance created");
        FUNC_EXIT();
    }
    int* MoveDecision(const std::vector<std::vector<int>> &grid, int gold, int gold_2)
    {
        FUNC_ENTRY();
        LOG("=== MoveDecision called with gold=" + std::to_string(gold) + ", gold_2=" + std::to_string(gold_2) + ", round=" + std::to_string(round_now + 1) + " ===");

        _mm_prefetch(nxt_op, _MM_HINT_T0);
        static int res[3] = {4, 4, 4};

        LOG("Converting grid to int16");
        FastSolve::cvt2int16(grid);
        const int16_t* ptr = reinterpret_cast<int16_t*>(FastSolve::res);

        LOG("Finding player and enemy positions");
        FastSolve::find_positions_avx2(ptr);

        LOG("Checking overlap");
        FastSolve::overlap_positive_sum_gt10(FastSolve::res);
        div_t pos_res = div(FastSolve::player_idx, map_size);

        LOG("Checking fast path with overlap_flag=" + std::string(FastSolve::overlap_flag ? "true" : "false"));
        if(FastSolve::overlap_flag && fast_check({pos_res.quot, pos_res.rem}, FastSolve::res)) {
            LOG("Using fast path prediction: [" + std::to_string(pred_res[0]) + "," + std::to_string(pred_res[1]) + "," + std::to_string(pred_res[2]) + "]");
            FUNC_EXIT();
            return pred_res;
        }

        LOG("Trying FastSolve best3step");
        auto fast_res = FastSolve::best3step_entry();
        if(fast_res.second != -1) {
            res[0] = reflect(fast_res.second & 3);
            res[1] = reflect((fast_res.second >> 2) & 3);
            res[2] = reflect((fast_res.second >> 4) & 3);
            LOG("Using FastSolve result: [" + std::to_string(res[0]) + "," + std::to_string(res[1]) + "," + std::to_string(res[2]) + "]");
            FUNC_EXIT();
            return res;
        }

        round_now++;
        LOG("Entering full search mode for round " + std::to_string(round_now));

        init_map(grid, new_grid, gold, round_now);
        last_coin = gold;
        int events = 0;
        if(!have_run) {
            LOG("First run initialization");
            have_run = true;

            pw_decay[0] = 1.0;
            for(int i = 1; i < 32; i++) {
                pw_decay[i] = pw_decay[i - 1] * decay_factor;
            }
            flag_npc_area(grid);
        } else {
            if(!npc_appear_round && round_now > 1) {
                check_npc(prev_map, grid, round_now);
            }
        }

        Position player, enemy;
        find_player(grid, player, enemy);

        LOG("Running full best move search");
        get_best_move(player, enemy, new_grid, res, round_now);

        LOG("Final decision: [" + std::to_string(res[0]) + "," + std::to_string(res[1]) + "," + std::to_string(res[2]) + "]");
        get_next_move(player, enemy, new_grid, res, round_now);
        prev_map = grid;
        LOG("=== MoveDecision completed ===");
        FUNC_EXIT();
        return res;
    }

    static Player& getInstance()
    {
        FUNC_ENTRY();
        static Player instance;
        LOG("Player singleton instance accessed");
        FUNC_EXIT();
        return instance;
    }

private:
    std::pair<int, int> position;
    int gold;
    int action;
    int data[3] = {4, 4, 4};
};

extern "C" {
    int* moveDecision(const std::vector<std::vector<int>> &grid, int gold, int gold_2) {
        // 初始化日志系统（只在第一次调用时）
        static bool log_initialized = false;
        if (!log_initialized) {
            DebugTracker::initLogFile("cpp_debug.log");
            log_initialized = true;
        }

        FUNC_ENTRY();
        auto result = Player::getInstance().MoveDecision(grid, gold, gold_2);
        FUNC_EXIT();
        return result;
    }
}

extern "C" void move_decision_c(const int32_t* grid_flat, int rows, int cols,
    int gold, int gold_2, int32_t out_moves[3]) {
    // 初始化日志文件
    static bool log_initialized = false;
    if (!log_initialized) {
        // 检查环境变量来配置日志
        const char* log_level_env = std::getenv("CPP_LOG_LEVEL");
        if (log_level_env) {
            DebugTracker::setLogLevel(std::atoi(log_level_env));
        }

        const char* log_console_env = std::getenv("CPP_LOG_CONSOLE");
        if (log_console_env && std::string(log_console_env) == "1") {
            DebugTracker::setConsoleOutput(true);
        }

        const char* log_file_env = std::getenv("CPP_LOG_FILE");
        std::string log_file = log_file_env ? log_file_env : "move_decision.log";

        DebugTracker::initLogFile(log_file);
        log_initialized = true;
    }

    FUNC_ENTRY_CRITICAL();  // 这是频繁调用的关键函数

    // 安全性检查（可选）
    if (rows != 17 || cols != 17) {
        LOG_CRITICAL("Invalid grid size: " + std::to_string(rows) + "x" + std::to_string(cols));
        // 简单回退：不合法尺寸时原地不动
        out_moves[0] = out_moves[1] = out_moves[2] = 4; // 你的默认值
        FUNC_EXIT_CRITICAL();
        return;
    }

    // 重建 Map
    Map grid(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            grid[i][j] = static_cast<int>(grid_flat[i * cols + j]);
        }
    }

    // 复用你已有的逻辑
    int* r = moveDecision(grid, gold, gold_2);
    out_moves[0] = static_cast<int32_t>(r[0]);
    out_moves[1] = static_cast<int32_t>(r[1]);
    out_moves[2] = static_cast<int32_t>(r[2]);

    LOG_CRITICAL("Generated moves: " + std::to_string(out_moves[0]) + ", " +
        std::to_string(out_moves[1]) + ", " + std::to_string(out_moves[2]));

    FUNC_EXIT_CRITICAL();

    // 注意：这里不能调用 closeLogFile()，因为函数可能被多次调用
}