__global__ void bilateralOutfocusing(float *out, float *in, float *depth, float *depthDiff, int w, int h, float sigma_d, float threshold0, float threshold1, float threshold2, float threshold3, float threshold4, float threshold5, float threshold6, float threshold7)
{
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= w || py >= h) {
    return;
  }
  float sum = 0.0f;
  float weight = 0.0f;
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  int idx = (py*w + px);
  float dp = depth[py*w + px];
  float dq;
  int idx_rgb = idx * 3;
  
  float diff = depthDiff[py*w + px];
  //int sigma = sigmaByDepth(log(diff), threshold0, threshold1, threshold2, threshold3, threshold4, threshold5, threshold6, threshold7);
  int max_k = 8;
  float sigma_dof = 10000.0f;
  float sigma_f = max_k * (1 - exp(-(diff*diff) / (sigma_dof*sigma_dof)));
  int sigma = floor(sigma_f);


  for (int i = -sigma; i <= sigma; i++) {
    for (int j = -sigma; j <= sigma; j++) {
      int qx = px + j;
      int qy = py + i;
      if (qx < 0 || qy < 0 || qx >= w || qy >= h)
        continue;
      dq = depth[qy*w + qx];
      weight = weightByDepth(sigma_f, i) * weightByDepth(sigma_f, j) * euclideanWeight(dp, dq, sigma_d);
      sum += weight;
      int new_idx = (qy*w + qx) * 3;
      float r_q = in[new_idx];
      float g_q = in[new_idx + 1];
      float b_q = in[new_idx + 2];
      float gray = 0.299*r_q + 0.578*g_q + 0.114*b_q;
      if (gray >= 0.98)
      {
        if (i*i + j*j <= sigma*sigma) {
          r_q *= 3;
          g_q *= 3;
          b_q *= 3;
        }
      }
      r += weight * r_q;
      g += weight * g_q;
      b += weight * b_q;
    }
  }
  out[idx_rgb] = r / sum;
  out[idx_rgb + 1] = g / sum;
  out[idx_rgb + 2] = b / sum;
}