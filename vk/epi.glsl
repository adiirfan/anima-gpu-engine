// Shared ending of the multiplies: what happens to output r of token n, output channel m.
//   EPI 0: YH[n][m] = r (fp16)     EPI 1: YH = gelu(r) (exact erf form)     EPI 2: RES[n][m] += r * GATE[m]
//   EPI 3: RES[n][m] = r           EPI 6: YH = gelu(r) (tanh form: Wan, Krea)
//   (+ BIAS[m] first when p.cen != 0: the centring offset W.mu, or a real bias)
float erf_(float x) {   // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7 (GLSL has no erf)
    const float s = sign(x); x = abs(x);
    const float t = 1.0 / (1.0 + 0.3275911 * x);
    const float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return s * y;
}
void epi(uint n, uint m, float r) {
    if (p.cen != 0u) r += BIAS[m];
    if (EPI == 0) YH[n * p.ldy + m] = float16_t(r);
    else if (EPI == 1) YH[n * p.ldy + m] = float16_t(0.5 * r * (1.0 + erf_(r * 0.70710678)));
    else if (EPI == 6) YH[n * p.ldy + m] = float16_t(0.5 * r * (1.0 + tanh(0.7978845608 * (r + 0.044715 * r * r * r))));
    else if (EPI == 2) RES[n * p.ldy + m] += r * GATE[m];
    else RES[n * p.ldy + m] = r;
}
