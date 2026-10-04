# DEFECTTWIN_2D — STAGE 2B

## الهدف

مقارنة تمثيلات بنيوية بسيطة قبل أي GNN:

1. metadata only
2. defect composition
3. defect graph descriptors
4. pristine graph descriptors
5. concatenated pristine + defect graph
6. twin difference graph descriptors

## ما المقصود بـgraph descriptors؟

يُبنى neighbor graph من:

- الأعداد الذرية
- الإحداثيات
- الخلية
- دورية x وy

ثم تُستخرج:

- degree statistics
- normalized degree histogram
- edge-distance statistics
- radial histogram
- homonuclear-edge fraction
- atomic-number contrast on edges
- composition and cell-density statistics

## منع تسرب الهدف

لا تستخدم المرحلة أيًا من:

- `magmom`
- `magmoms`
- `energy`
- `forces`

كميزات.

تُستخدم `json_magmom` فقط لبناء الـlabel الذي تم تثبيته في Stage 2A FIX1.

## التقييم

- Leave-One-Host-Out عبر 30 host.
- Binary magnetic activation.
- Four-level ordinal magnitude.
- نفس Random Forest ونفس الأوزان لكل feature set.
- CPU وOffline بالكامل.

## التشغيل

فكّي الحزمة في:

`D:\D2_STAGE2B`

ثم شغّلي:

`RUN_PROJECT.bat`

بعد النهاية ارفعي:

`STAGE2B_RESULTS.zip`
