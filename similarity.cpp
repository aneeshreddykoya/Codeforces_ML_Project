#include <cmath>

extern "C" {
    // Computes cosine similarity between two double arrays of size n
    double cosine_similarity_c(double* vec1, double* vec2, int n) {
        double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
        for(int i = 0; i < n; ++i) {
            dot += vec1[i] * vec2[i];
            denom_a += vec1[i] * vec1[i];
            denom_b += vec2[i] * vec2[i];
        }
        if (denom_a == 0 || denom_b == 0) return 0.0;
        return dot / (sqrt(denom_a) * sqrt(denom_b));
    }
}
