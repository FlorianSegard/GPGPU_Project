

/*
__device__ bool has_changed;

__global__ void hysteresis_reconstruction(const lab* input, bool* marker, bool* output, int width, int height, std::ptrdiff_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    lab* lineptr;
    lineptr = (lab*)((std::byte*)input + y * stride);
    int current_idx = y * width + x;

    if (output[current_idx] || !marker[current_idx]) // already processed or too low
        return;

    // Check 8-connected neighbors
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int neighbor_idx = ny * width + x;
                if (output[neighbor_idx]) {
                    output[current_idx] = true;
                    has_changed = true;
                    return;
                }
            }
        }
    }
}
*/
bool has_changed;

void hysteresis_reconstruction(const lab* input, bool* marker, bool* output, int width, int height, std::ptrdiff_t stride) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int current_idx = y * width + x;
            if (output[current_idx] || !marker[current_idx])
                continue;

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int neighbor_idx = ny * width + x;
                        if (output[neighbor_idx]) {
                            output[current_idx] = true;
                            break;
                        }
                    }
                }
            }
        }
    }
}