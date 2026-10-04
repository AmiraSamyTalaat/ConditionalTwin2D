# Next confirmatory lockbox for Strong Q1

## الهدف

اختبار ConditionalTwin-2D بعد تجميده على مصدر لم تُفتح Magnetic targets الخاصة به أثناء تصميم الطريقة.

## الخيار الأفضل

Prospective DFT mini-lockbox:

- 5-8 hosts جديدة.
- 20-30 عيبًا لكل host.
- 100-200 سجل إجمالي.
- Vacancies وsubstitutions مع توزيع محدد مسبقًا.
- Relaxed structures وtotal/local magnetic moments.

## التسلسل الإلزامي

1. تجميد extractor والـ68 features وطريقة host-centering.
2. تجميد seeds والنموذج والعتبة.
3. تجميد قائمة hosts والعيوب.
4. توليد التصنيف والتنبؤ المستمر.
5. حفظ prediction seal وSHA-256.
6. تنفيذ/فتح DFT targets بعد الختم فقط.
7. الإبلاغ عن كل host وكل record.

## المقاييس الأولية

1. Within-host pair AUROC.
2. Macro-host AUROC.
3. Worst-host AUROC مع حد أدنى منطقي لحجم الـhost.
4. Pooled AUROC.
5. Top-10% وTop-20% within-host enrichment.
6. Calibration وBrier skill.
7. Continuous host-centered Spearman وMAE.

## ممنوع بعد فتح النتائج

- External recalibration.
- تغيير threshold.
- حذف host ضعيف.
- اختيار subset ناجح.
- إعادة اختيار features.
- score inversion.
- تغيير طريقة centering.

## الحكم

هذه التجربة هي العنصر الذي يمكنه تحويل الجدة الحالية إلى محاولة Strong Q1 حقيقية. النتائج السابقة لا يمكن إعادة استخدامها كاختبار تأكيدي لأنها أصبحت معروفة.
