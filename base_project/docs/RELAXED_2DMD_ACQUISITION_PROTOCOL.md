# Relaxed 2DMD structure audit - frozen diagnostic protocol

## لماذا هذه التجربة مهمة؟

النسخة المعالجة من 2DMD توفر `initial.tar.gz` ببنى CIF غير مسترخية، بينما الخصائص حُسبت بعد structural relaxation. لذلك يجب اختبار ما إذا كان استخدام البنية المسترخية يغير استنتاجات النقل.

هذه التجربة أصبحت **diagnostic** وليست confirmatory، لأن نتائج 2DMD الحالية معروفة بالفعل.

## المصدر الرسمي

Repository:

```text
https://github.com/HSE-LAMBDA/ai4material_design
```

Dataset documentation:

```text
https://github.com/HSE-LAMBDA/ai4material_design/blob/main/docs/DATA.md
```

## أوامر DVC الرسمية

نفذيها على قرص لديه مساحة كبيرة:

```bash
git clone https://github.com/HSE-LAMBDA/ai4material_design.git
cd ai4material_design
python -m pip install "dvc[s3]"
dvc pull -R datasets/raw_vasp/high_density_defects datasets/raw_vasp/dichalcogenides8x8_vasp_nus_202110
```

المصدر الرسمي يحذر من أن بعض الملفات مضغوطة وأن الحجم بعد فكها يقارب 300 GB. لا تشغلي هذه الخطوة على قرص محدود المساحة.

## التجميد قبل قراءة النتائج

1. ثبتي قائمة `_id` المطابقة لسجلات 2DMD المستخدمة في الحزمة الحالية.
2. احفظي القائمة وSHA-256 قبل استخراج أي structure نهائي.
3. لا تغيري الـhosts أو العتبة أو قائمة السجلات بعد رؤية الأداء.
4. استخرجي آخر ionic structure صالحة من `CONTCAR` أو آخر ionic step موثق.
5. سجلي أي سجل مفقود أو غير متقارب؛ لا تستبدليه بسجل آخر ناجح.

## المقارنة المطلوبة

لنفس السجلات والنموذج المجمد:

```text
Initial-structure 68 features
versus
Relaxed-structure 68 features
```

المقاييس الأساسية:

- Within-host pair AUROC.
- Host-centered AUROC.
- Macro-host AUROC.
- Pooled AUROC.
- Continuous host-centered Spearman.
- Brier/NLL/ECE.
- Per-host paired differences.

## بوابات التحقق

- سجل واحد إلى واحد بين `_id` والبنية النهائية.
- عدم فتح أو استخدام labels لاختيار طريقة parsing.
- feature extractor نفسه دون تعديل.
- hashes للبنى والـfeatures.
- paired host bootstrap.
- الاحتفاظ بكل النتائج سواء تحسنت أم لم تتحسن.
