# Матрица методов выбора контекста

Дата отсечения — 8 ноября 2026 года. Таблицы сравнивают **постановки оригинальных работ**. Причинная адаптация, наша реимплементация и повторение авторского benchmark не считаются одним уровнем свидетельств. Подробные метаданные и ограничения находятся в `sources.json`.

В нашей записи K — число строк контекста. В статье CRUMB K обозначает число query-кластеров, а длина контекста обозначена иначе. Эти бюджеты нельзя механически отождествлять.

## Контекст табличной модели

| Метод | Выбираемый объект | Единица контекста | Критерий оригинала | Адаптация и стоимость | Прямой uplift scope |
|---|---|---|---|---|---|
| [Nagler 2023](https://proceedings.mlr.press/v202/nagler23a/nagler23a.pdf) | Реальные строки | Запрос | Локализация по X | Retrieval до prediction | Нет CATE-selector |
| [LoCalPFN](https://arxiv.org/html/2406.05207v1) | Реальные строки | Запрос | kNN и локальная адаптация | Полный метод включает fine-tuning | Tabular prediction |
| [LE-TabPFN](https://openreview.net/pdf?id=LFyQyV5HxQ) | Строки в embedding | Локальный запрос | Близость в представлении | Стоимость representation отдельно | Tabular prediction |
| [MixturePFN/MICP](https://proceedings.iclr.cc/paper_files/paper/2025/file/3de60dc2ff7ac8866e805bbf457e2291-Paper-Conference.pdf) | Локальные наборы и routing | Группа запросов | Локальность и общая память | Отделять routing от CaPFN | Tabular prediction |
| [CRUMB](https://arxiv.org/html/2606.11473v1) | Реальные строки | Query cluster | MMD с target X | Clustering, herding, shared inference | Causal adaptation нужна отдельно |
| [VIP-COP](https://arxiv.org/html/2605.12904v2) | Реальные строки и признаки | Общий набор | Validation performance FM | Многократные пробные forward | Causal reward — адаптация |
| [ICD](https://arxiv.org/html/2402.06971v1) | Синтетический контекст | Компактный prompt | Distillation | Оптимизация искусственных данных | Не factual-row selector |
| [TuneTables](https://arxiv.org/html/2402.11137v3) | Обучаемый prompt | Общий компактный контекст | Supervised quality | Fitting и gradient access | Не uplift-selector |
| [LUCoS](https://arxiv.org/html/2605.27254v1) | Кандидаты до Y | Cold-start набор | Latent representation | Сначала выбор, затем annotation | Label acquisition |
| [Tab-AICL](https://arxiv.org/html/2603.27385v1) | Неизвестные labels | Растущий контекст | Active acquisition | Учитывать купленные Y | Label acquisition |
| [EPIC/TANN](https://arxiv.org/html/2611.07886v1) | Неизвестные labels | Контекст frozen TFM | Target-aware acquisition | Обновление контекста, acquisition cost | Predictive, не causal reward |
| [ARASH](https://arxiv.org/html/2608.17856v1) | Строки и shot budget | Запрос | Adaptive retrieval | K и обходы FM входят в цену | Tabular prediction |
| [TAAR](https://openreview.net/pdf?id=qBMNsGiE3u) | Реальные строки | Target-dependent | Attention alignment | Нужны model-aware сигналы | Анонимный primary PDF |
| [Test-Time Compute](https://arxiv.org/html/2610.12005v1) | Контекст и операции inference | Несколько режимов | Model-aware quality/compute | Считать все адаптационные вызовы | Tabular prediction |
| [TabDPT-Turbo](https://arxiv.org/html/2608.01400v2) | Архитектура чтения таблицы | Большой контекст | Эффективный inference | Не отдельный subset-selector | Граница вычислительной мотивации |

## Причинные модели и выбор данных

| Метод | Что известно при выборе | Что меняется | Цель | Основной бюджет | Граница переноса |
|---|---|---|---|---|---|
| [CausalPFN](https://arxiv.org/html/2506.07918v2) | Support X/T/Y и query X | Native weak-effect context | CEPO и treatment effects | Контекст, weak fitting, inference | Нативный retrieval уже существует |
| [Do-PFN](https://arxiv.org/html/2506.06039v3) | Observational context | Posterior под SCM prior | Причинные величины | Pretraining и контекст | Документированы continuous X/Y |
| [CausalFM](https://arxiv.org/html/2506.10914v3) | Support и setting assumptions | Setting-specific model | Causal targets | Prior, context, inference | Проверять идентификационную постановку |
| [TabPFN-3 meta-learners](https://arxiv.org/html/2605.13986v3) | Supervised causal dataset | T/S/X procedure | Uplift и отдельные CATE задачи | Fitting/context всего pipeline | Это уже прямой uplift benchmark |
| [CInA](https://proceedings.mlr.press/v235/zhang24x.html) | Наблюдаемые causal datasets | Balancing/attention | Причинное оценивание | Meta-training | Не discrete K-row selector |
| [QHTE](https://proceedings.mlr.press/v139/qin21b.html) | X/T и купленные Y | Acquire и refit | CATE/PEHE | Число новых outcomes | Нельзя приравнивать к размеченной памяти |
| [Causal-BALD](https://proceedings.neurips.cc/paper_files/paper/2021/file/ffa4eb0e32349ae57f7a0ee8c7cd7c11-Paper.pdf) | Pool и posterior | Outcome acquisition | Treatment-effect information | Labels и модель | Uncertainty не равна policy value |
| [Causal-EPIG](https://proceedings.mlr.press/v306/gao26i.html) | Target X и posterior | Приобретение outcomes | Информация о CATE/PO | Label budget | Уже известен causal-objective alignment |
| [GATE](https://openreview.net/pdf?id=2Q3gFNbpAr) | Observational few-shot data | Генерация контрфактических labels | Downstream CATE | Генерация и обучение | Workshop; иной inference path |
| [MALTS](https://jmlr.org/papers/v23/21-0053.html) | Обучающие X/T/Y | Learned distance | Causal matching | Обучение метрики и соседи | Гарантии не наследуются transformer |
| [Causal optimal transport](https://arxiv.org/abs/2109.01991) | Observational data | Веса matching | Causal balance | Вычисление весов | Весовой estimator не равен raw subset |
| [CFR / Shalit et al. 2017](https://proceedings.mlr.press/v70/shalit17a.html) | Наблюдаемые X/T/Y | Learned representation | Factual loss и IPM между arms | Обучение estimator | Это не MMD context/target selection |
| [Kato & Kato 2026](https://arxiv.org/abs/2607.18225) | Action-specific historical evidence | RAG/matching для действий | Policy regret | Retrieval и generator inference | Не проверенный learned swap selector causal PFN |

## Обучение utility и решение о treatment

| Работа | Объект оптимизации | Почему близка | Что не закрывает |
|---|---|---|---|
| [EPR](https://aclanthology.org/2022.naacl-main.191/) | Retrieval для NLP | Отклик frozen model как supervision | Идентификацию causal reward |
| [CEIL](https://proceedings.mlr.press/v202/ye23c.html) | Композиция exemplars | Полезность набора | Uplift-policy estimator |
| [Iterative retrieval](https://aclanthology.org/2024.emnlp-main.406/) | Последовательный набор | Marginal improvement по model feedback | Новизну простой causal-адаптации |
| [Distributional Shapley](https://proceedings.mlr.press/v119/ghorbani20a.html) | Ценность новых точек | Амортизация valuation через regression | Наш конкретный causal estimand |
| [NASH](https://proceedings.mlr.press/v306/tian26c.html) | Selection по data values | Ограничения линейного top-m | Автоматическое superiority на FM |
| [Localizing Global Discrepancies](https://arxiv.org/abs/2608.28375) | Fixed-size replacement contribution | Оператор ожидаемой замены | Identification без causal labels |
| [Rank-Learner](https://proceedings.mlr.press/v306/arno26a.html) | Treatment-effect ranking | Ортогональная целевая функция | Выбор donor-контекста |
| [Empirical welfare maximization](https://tetenov.com/Who_should_be_treated_Kitagawa_Tetenov.pdf) | Ограниченная treatment-policy | Бюджет и welfare | Конкретную механику FM |
| [RATE](https://pmc.ncbi.nlm.nih.gov/articles/PMC12002561/) | Оценка prioritization rules | Честная ranking inference | Гарантию обученного на test selector |

## Ближайшие проверки критики

Это вопросы нашего обзора, а не заранее полученные отрицательные результаты.

| Семейство | Что сохраняем при сравнении | Проверяемое ограничение | Благоприятный контроль |
|---|---|---|---|
| Native CausalPFN | Weak learner, retrieval union, batch rules | Ошибка teacher и зависимость от batch | Точный teacher или простой эффект |
| Feature-kNN | Одинаковые encoding/scaling/K | Нерелевантные признаки и геометрия эффекта | Низкая размерность, локальная гладкость |
| CRUMB | Query clustering и batching advantage | Matching X против causal decision utility | Исправимый covariate shift |
| VIP-COP | Тот же causal reward и offline compute | Global/additive valuation против query-conditioned utility | Почти аддитивный полезный набор |
| Acquisition | Корректный доступ к неизвестным Y | Перенос в другую задачу labelled memory | Исходный label-budget режим |
| Новый swap-кандидат | Anchors, arm counts, honest training | Непредсказуемая utility, donor/cohort shift и interactions | Стабильный перенос utility при ограниченных заменах |

Все методы оцениваются на общей итоговой политике. Для каждого выбранного контекста отдельно считать direct-local, а для обучаемого selector — scorer/teacher без FM. Если меняется качество вместе с representation, нормировкой, числом вызовов или доступом к labels, это отдельная ось эксперимента.

## Текущая готовность проекта

В `../new_context_method/crumb.py` есть реимплементация алгоритма CRUMB по статье с записанными допущениями. В `baselines.py` отдельно находятся random, feature-kNN, scalar-effect kNN и treatment-stratified MMD control. Последний не выдаётся за исходный CRUMB. Нативный CausalPFN доступен в закреплённом окружении и исторических runners.

VIP-COP с causal reward, learned retriever и новый utility predictor ещё не обучены и не сравнены. Численные проверки и маленький CPU smoke подтверждают вычисление правил и передачу контекста, но не повторяют авторские benchmarks и не показывают эффективность нового метода.
