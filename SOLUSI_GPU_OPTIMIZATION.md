# 🔧 SOLUSI: GPU Usage Rendah (<5%) pada Training

## **MASALAH UTAMA**
Training sudah menggunakan GPU, tetapi GPU utilization hanya 5% → bottleneck data loading & batch size terlalu kecil.

---

## **PENYEBAB & SOLUSI**

### **1. Tingkatkan `num_workers` (Data Loading)**
**Baris 354** di notebook:
```python
# ❌ SEBELUMNYA:
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, drop_last=True, pin_memory=torch.cuda.is_available(),
)

# ✅ SESUDAHNYA:
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=8, drop_last=True, pin_memory=torch.cuda.is_available(),
)
```
**Alasan:** 2 workers terlalu sedikit untuk RTX 4070 SUPER. Gunakan 8-16 workers agar data prefetch lebih cepat.

---

### **2. Naikkan Batch Size (GPU Utilization)**
**Baris 131** di konfigurasi:
```python
# ❌ SEBELUMNYA:
BATCH_SIZE   = 32

# ✅ SESUDAHNYA (untuk RTX 4070 dengan 12GB VRAM):
BATCH_SIZE   = 64  atau 96 bahkan 128
```
**Alasan:** RTX 4070 SUPER punya 12GB VRAM → bisa handle batch size 64-128 untuk IMG_SIZE=64.

---

### **3. Optimize R1 Penalty (Gradient Penalty)**
**Baris 140** di konfigurasi:
```python
# ❌ SEBELUMNYA:
R1_EVERY     = 16  # terapkan R1 setiap 16 step

# ✅ SESUDAHNYA:
R1_EVERY     = 32 atau 64  # kurangi frekuensi R1 penalty
```
**Alasan:** R1 penalty sangat computationally expensive. Terapkan setiap 32-64 step (bukan 16) agar training lebih cepat tanpa mengurangi kualitas.

---

### **4. Gunakan Mixed Precision (Opsional tapi POWERFUL)**
Tambahkan di **bagian training loop (sebelum `for epoch in range...`)**:
```python
# Gunakan automatic mixed precision untuk speedup 20-30%
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(1, EPOCHS + 1):
    for real_imgs, labels in dataloader:
        real_imgs = real_imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        bs = real_imgs.size(0)

        # ✅ WRAP discriminator update dengan autocast
        with autocast():
            z = torch.randn(bs, LATENT_DIM, device=DEVICE)
            fake_imgs = G(z, labels).detach()
            real_logits = D(real_imgs, labels)
            fake_logits = D(fake_imgs, labels)
            d_loss = hinge_loss_dis(real_logits, fake_logits)
            
            if global_step % R1_EVERY == 0:
                r1 = r1_penalty(D, real_imgs, labels)
                d_loss = d_loss + (R1_GAMMA / 2) * r1
        
        opt_D.zero_grad(set_to_none=True)
        scaler.scale(d_loss).backward()
        scaler.step(opt_D)

        # ✅ WRAP generator update dengan autocast
        with autocast():
            z = torch.randn(bs, LATENT_DIM, device=DEVICE)
            fake_imgs = G(z, labels)
            fake_logits, fake_feats = D(fake_imgs, labels, return_features=True)
            with torch.no_grad():
                _, real_feats = D(real_imgs, labels, return_features=True)
            
            g_adv_loss = hinge_loss_gen(fake_logits)
            fm_loss = classwise_feature_matching_loss(real_feats, fake_feats, labels, NUM_CLASSES)
            g_loss = g_adv_loss + FM_LAMBDA * fm_loss
        
        opt_G.zero_grad(set_to_none=True)
        scaler.scale(g_loss).backward()
        scaler.step(opt_G)
        scaler.update()
        
        ema.update(G)
        global_step += 1
```

---

## **PERBANDINGAN OPTIMASI**

| Konfigurasi | Batch Size | Num Workers | GPU Util | Speed | VRAM |
|---|---|---|---|---|---|
| **SEBELUMNYA** | 32 | 2 | ~5% | Slow | ~2GB |
| **Rekomendasi 1** | 64 | 8 | ~30-50% | 2x lebih cepat | ~4GB |
| **Rekomendasi 2** | 96 | 12 | ~50-70% | 3x lebih cepat | ~6GB |
| **Rekomendasi 3 (Max)** | 128 | 16 | ~70-90% | 4x lebih cepat | ~8GB |
| **+ Mixed Precision** | 128 | 16 | ~80-95% | 5-6x lebih cepat | ~4-5GB |

---

## **LANGKAH IMPLEMENTASI CEPAT**

### **1. Update file (Minimal)**
Hanya ubah 3 baris di konfigurasi:
- Line 131: `BATCH_SIZE = 64` 
- Line 140: `R1_EVERY = 32`
- Line 354: `num_workers=8`

### **2. Testing First**
Jalankan 1-2 epoch untuk cek:
```bash
# Di notebook: ubah EPOCHS = 2 dulu
# Jalankan training loop
# Monitor GPU usage dengan: nvidia-smi -l 1  (di terminal baru)
```

### **3. Monitor GPU**
Di terminal baru, run real-time monitor:
```bash
nvidia-smi -l 1
# atau
watch -n 1 nvidia-smi
```
Seharusnya GPU Util naik ke **30-50%** (atau lebih dengan mixed precision).

---

## **EXPECTED RESULTS**

**Sebelum Optimasi:**
- GPU Util: ~5%
- 1 epoch banana_ripeness (500+ images): ~30 detik
- 500 epochs: ~4 jam

**Setelah Optimasi (Batch=64, Workers=8, R1_EVERY=32):**
- GPU Util: ~40-60%
- 1 epoch: ~10-15 detik
- 500 epochs: ~1-2 jam ✅

**Setelah Optimasi + Mixed Precision:**
- GPU Util: ~70-90%
- 1 epoch: ~5-8 detik
- 500 epochs: ~30-50 menit ✅✅

---

## **CATATAN PENTING**
- ✅ Model **SUDAH menggunakan GPU** (DEVICE=cuda, model.to(DEVICE) aktif)
- ✅ `pin_memory=True` sudah benar (membantu transfer data CPU→GPU)
- ⚠️ **Penyebab rendahnya GPU util = data loading bottleneck + batch size kecil**
- ✅ Solusi: Naikkan num_workers dan batch size, aplikasikan mixed precision
