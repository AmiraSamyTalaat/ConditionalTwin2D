# DefectTwin2D - ConditionalTwin-2D

## الحكم التنفيذي

هذه الحزمة لا تخفي نتيجة QPOD السلبية ولا تعيد ضبط النموذج عليها. بدلًا من ذلك، تضيف طريقة جديدة مصممة لترتيب العيوب **داخل الـhost نفسه** وتقليل النجاح المضلل الناتج عن اختلاف متوسطات الـhosts.

الطريقة النهائية اسمها **ConditionalTwin-2D**. وهي تجمع:

- الخصائص الـ68 الأصلية المجمدة.
- 68 انحرافًا label-blind عن متوسط مجموعة العيوب المرشحة داخل الـhost نفسه.
- وزنًا كليًا متساويًا لكل host.
- موازنة للفئتين داخل كل host مختلط.

اختيار الطريقة تم من Development فقط. لم تستخدم Labels من 2DMD أو Imp2D أو QPOD في بناء التمثيل أو اختيار النموذج.

## أقوى نتيجة جديدة

| المقياس على Development LOHO | ExtraTrees | ConditionalTwin-2D |
|---|---:|---:|
| Pooled AUROC | 0.9079 | 0.9318 |
| Macro-host AUROC | 0.7957 | 0.8565 |
| Within-host pair AUROC | 0.8089 | 0.8608 |
| Host-centered AUROC | 0.7297 | 0.7166 |
| Shortcut amplification index | 0.0990 | 0.0711 |

التحسن المقترن في within-host AUROC:

```text
Mean difference = +0.0500
95% host-bootstrap CI = [+0.0214, +0.0860]
P(difference > 0) = 0.993
```

احتفظ تحليل حساسية حذف صفوف الـhost بالفرع `Conditional_only` في جميع الجداول الثلاثين المتداخلة، كما اختاره جدول Development الكامل. هذه الجداول تعيد تقييم تنبؤات cross-fitted ثابتة وليست تكرارات مستقلة ولا nested outer-fold estimates. تم رفض فرعي host-prevalence وlinear PairRank بدل الاحتفاظ بهما لزيادة التعقيد شكليًا.

## Ablations

| التجربة | Within-host AUROC |
|---|---:|
| ExtraTrees الأصلي | 0.8089 |
| تغيير host/class weights فقط | 0.8201 |
| Host-centered descriptors فقط | 0.8431 |
| Raw + centered + host/class weights | **0.8608** |

هذا يفصل أثر إعادة الوزن عن أثر التمثيل الشرطي.

## النتائج الخارجية السابقة

هذه النتائج **retrospective diagnostics** وليست validation جديدة:

- 2DMD: ارتفع within-host AUROC من 0.4774 إلى 0.5673، بينما انخفض pooled AUROC من 0.7216 إلى 0.5441. هذا يتفق مع إزالة جزء من between-host shortcut، وليس مع تفوق شامل.
- Imp2D: لا يوجد تحسن مقنع.
- QPOD: الأداء قريب جدًا من ExtraTrees ولا يثبت نقلًا جديدًا.

## القيد التطبيقي

ConditionalTwin-2D تحتاج مجموعة من العيوب المرشحة للـhost نفسه كي تحسب المتوسط label-blind. لذلك هي **candidate-pool conditional** وليست نموذجًا inductive لسجل منفرد بلا سياق.

## هل أصبح البحث Strong Q1؟

البحث أصبح أقوى خوارزميًا ومنهجيًا من النسخة السابقة، لأن لديه الآن:

- طريقة جديدة تستهدف host shortcuts مباشرة.
- leave-one-host-out cross-fitted finite selection.
- ablations صريحة.
- paired host bootstrap.
- screening utility.
- transport matrix عبر ثلاثة cohorts خارجية سابقة.
- نتائج سلبية محفوظة دون حذف أو external tuning.

لكن محاولة Strong Q1 ما زالت تحتاج **lockbox جديدة لم تُفتح نتائجها أثناء تصميم الطريقة** أو prospective DFT mini-lockbox. لا توجد صياغة أو إحصائية يمكنها استبدال هذا الدليل.

## الملفات المهمة

- `outputs/ARABIC_STAGE_REPORT.md`: تقرير علمي عربي.
- `outputs/MANUSCRIPT_TRANSPORTRANK_INSERT.md`: نص إنجليزي جاهز للدمج.
- `outputs/tables/DefectTwin2D_TransportRank_Tables.xlsx`: ملف Excel موحد.
- `outputs/tables/TABLE_PAIRED_BOOTSTRAP_DIFFERENCES.csv`: أهم اختبار للتحسن.
- `outputs/figures/`: الأشكال بصيغ PNG/PDF/SVG.
- `outputs/REVIEWER_COMPLETENESS_MATRIX.csv`: ما اكتمل وما يزال ناقصًا.
- `docs/RELAXED_2DMD_ACQUISITION_PROTOCOL.md`: بروتوكول تدقيق البنى المسترخية.
- `docs/NEXT_STRONG_Q1_LOCKBOX.md`: تصميم التجربة التأكيدية القادمة.

## التشغيل على Windows

شغّلي فقط:

```text
RUN_PROJECT.bat
```

الملف ينشئ بيئة Python محلية عند الحاجة، يثبت المتطلبات، يشغّل التحليل، ثم ينفذ التحقق المستقل.

للتحقق من النتائج الموجودة دون إعادة التشغيل:

```text
VERIFY_ONLY.bat
```

علامة النجاح:

```text
SUCCESS. Upload outputs\RESULT_PACKAGE.zip
```
