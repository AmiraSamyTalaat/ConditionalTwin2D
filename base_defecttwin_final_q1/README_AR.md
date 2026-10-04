# DefectTwin-2D - الحزمة النهائية بعد إصلاح Q1

## ماذا تحتوي الحزمة؟
- المخطوط النهائي والملحق بصيغتي DOCX وPDF.
- ملف Excel موحد.
- جميع الجداول CSV/LaTeX.
- جميع الرسومات PNG/PDF/SVG.
- التنبؤات المتداخلة والـbootstrap checkpoints.
- Reviewer Completeness Matrix.
- الأدلة الأصلية للـlockboxes.
- البيانات المعالجة والكود والإعدادات والتحقق.

## الإصلاحات الفعلية
- إلغاء Beta غير الصالحة والرجوع الصامت إلى Identity.
- Calibration متداخلة صحيحة لـIdentity/Platt/Beta/Temperature.
- Macro-host calibration وslope/intercept.
- 5,000 paired host bootstraps وWilcoxon + Holm.
- Threshold confidence intervals.
- Continuous target مع Median AE وhost-centered Spearman.
- Baselines صريحة بين الـhosts وداخلها.
- Applicability Domain من خمسة مكونات، بما فيها Isolation Forest وensemble uncertainty.
- Reliability diagrams كاملة.
- Excel موحد ومصفوفة اكتمال للمحكمين.
- مخطوط وملحق محدثان مع claims أضيق وصادقة.

## الحكم العلمي
الحزمة أقوى كثيرًا لمحاولة مجلة Q1 متخصصة، لكنها لا تضمن القبول.
الجدة المنهجية والصرامة قوية؛ الجدة الخوارزمية متوسطة.
فشل Imp2D والقيود داخل 2DMD موثقة ولا يجوز استخدامها لإعادة الضبط.

## قبل الإرسال
يضاف فقط:
1. Affiliation.
2. Corresponding-author email.
3. Repository/Zenodo DOI بعد الأرشفة.
4. اختيار المجلة وتطبيق قالبها في النهاية.

شغلي RUN_PROJECT.bat فقط للتحقق من سلامة الحزمة، وليس لإعادة تدريب lockboxes.
