# Track C: основания и ближайшие мосты к utility-swap selector

Проверка на 8 ноября 2026 года. Карточки 30 первичных работ находятся в `track_c_sources.json`; 27 полных текстов и три источника с ограниченным доступом к оригинальному тексту. Поиск целевой, с переходами по связанным работам, а не доказательство полноты. Отсутствие совпадения в этой выборке не устанавливает приоритет метода. `code_url: null` означает, что авторский код здесь не подтверждён. Коды не запускались.

## Что уже занято литературой

| Компонент будущего метода | Опора | Следствие для формулировки вклада |
|---|---|---|
| Каузальные группы соседей, обученная метрика | [MALTS](https://jmlr.org/papers/v23/21-0053.html), [Variable Importance Matching](https://proceedings.mlr.press/v216/lanners23a.html) | Само outcome-informed расстояние и честное разделение обучения/оценки известны. |
| Сжатие признаков в causal scores | [Propensity](https://doi.org/10.1093/biomet/70.1.41), [prognostic score](https://doi.org/10.1093/biomet/asn004), [двойной matching](https://doi.org/10.1111/biom.12887) | Два score или два потенциальных исхода не образуют самостоятельной новизны. |
| Представительность источника относительно цели | [Causal Optimal Transport](https://arxiv.org/abs/2109.01991) | Дистрибуционное согласование уже является causal-инструментом; задача конечного FM-контекста отличается. |
| Teacher-guided пары в uplift | [KDSM](https://arxiv.org/abs/2303.02980) | Реальные treated/control пары уже использованы для обучения student; это предшественник pairing, не retrieval замороженной FM. |
| Выбор по отклику замороженной модели | [EPR](https://aclanthology.org/2022.naacl-main.191/), [CEIL](https://proceedings.mlr.press/v202/ye23c.html) | И feedback, и query-зависимые наборы, и моделирование взаимодействий известны. |
| Последовательная полезность набора | [Active Example Selection](https://aclanthology.org/2022.emnlp-main.622/), [RetICL](https://arxiv.org/abs/2305.14502), [iterative retrieval](https://aclanthology.org/2024.emnlp-main.406/) | RL и marginal-gain targets сами по себе не являются новым механизмом. |
| Перенос retrieval между задачами/моделями | [UPRISE](https://aclanthology.org/2023.emnlp-main.758/) | Перенос на новый банк необходим как проверка, но недостаточен как заявка новизны. |

Эти мосты требуют раздельной трактовки областей. Гарантия causal matching не переносится автоматически на transformer, а провал causal-адаптации языкового retriever не опровергает исходную NLP-работу. Аналогично, получение обеих treatment-групп в контексте не обеспечивает overlap, калибровку или идентификацию эффекта.

## Самые близкие ограничения для utility-swap

[Distributional Shapley, C15](https://proceedings.mlr.press/v119/ghorbani20a.html) — обязательный предшественник амортизации: §3/Algorithm 2 уже регрессирует оценённые значения на характеристики точек и предсказывает ценность ранее не оценённых точек. Значит, «измерить вклад части строк, обучить дешёвый predictor, перенести на новые строки» — известный шаблон.

[Localizing Global Discrepancies, C23](https://arxiv.org/abs/2608.28375) — обязательный prior для ожидаемой замены при фиксированном размере. Это математическая конструкция для статистик и anomaly detection, а не готовый uplift-selector. Кандидату нужен специфический estimand и экспериментальный результат сверх этой конструкции.

[Iterative retrieval, C12](https://aclanthology.org/2024.emnlp-main.406/) — ближайший общий алгоритмический контроль: уже учит улучшения от изменения набора по отклику модели. Замена языкового reward на causal reward должна быть раскрыта как адаптация. Сильный контроль — тот же reward и тот же бюджет offline FM-вызовов.

[Data Shapley](https://proceedings.mlr.press/v97/ghorbani19c.html), [Beta Shapley](https://proceedings.mlr.press/v151/kwon22a.html) и [Data Banzhaf](https://proceedings.mlr.press/v206/wang23e.html) задают разные распределения фоновых коалиций и способы обращения с шумной полезностью. Ограничение размера требует сравнения с этими семействами, а не заявления, что вся такая постановка впервые возникла здесь.

[Priority-Aware Shapley, C22](https://arxiv.org/abs/2602.09326) показывает существование структурно ограниченной valuation. Его adjacent-swap sampler переставляет порядок игроков; это **не** замена donor в FM-контексте. Наши arm-квоты также не тождественны precedence constraints. Связь полезна для постановки, но это разные операторы и теоремы.

## Аддитивная оценка не равна оптимальному набору

[Rethinking Data Shapley, C20](https://proceedings.mlr.press/v235/wang24cg.html) и [NASH, C21](https://proceedings.mlr.press/v306/tian26c.html) делают обязательным контроль взаимодействий. Для нашего метода это означает измерять реализованную полезность выбранного набора: качество regression targets или сумма положительных swap scores недостаточны. NASH стоит включить как ближайшую альтернативу глобальной линейной агрегации, сохранив различие его supervised задачи и нашей causal задачи.

[DVRL](https://proceedings.mlr.press/v119/yoon20a.html) и [GLISTER](https://doi.org/10.1609/aaai.v35i9.16988) уже оптимизируют выбор данных по целевому качеству. Их retraining-бюджет и доступ к labels отличаются от замороженной FM. [Data-OOB](https://proceedings.mlr.press/v202/kwon23e.html) полезен как эффективный surrogate valuation, но его bagging-specific value нельзя объявлять точным FM-control.

Представительные [CRAIG](https://proceedings.mlr.press/v119/mirzasoleiman20a.html), [GRAD-MATCH](https://proceedings.mlr.press/v139/killamsetty21a.html), [kernel herding](https://proceedings.mlr.press/r8/chen10a.html) и [conditional-gradient связь](https://icml.cc/2012/papers/683.pdf) закрывают общую новизну coreset, gradient matching и MMD. В [constrained local search](https://research.google/pubs/non-monotone-submodular-maximization-under-matroid-and-knapsack/) feasible exchange уже стандартен. Наш reward пока не доказан submodular, поэтому соответствующие approximation ratios к нему неприменимы.

## Где остаётся проверяемый кандидат

Следующее — наше умозаключение, а не найденный литературный результат. Узкий объект исследования: query-conditioned utility **допустимой замены реальных donor-строк** при fixed K/arm-квотах, причём target учитывает вытесненных из treatment-бюджета людей во всей meta-когорте; predictor переносится на новые query-группы и donor bank; на deployment нужен один итоговый FM-вызов.

Эта совокупность ещё не подтверждена как новая. Проверяемые отличия: whole-cohort budget displacement против локального reward; условная swap utility против глобальной row value; перенос вне обучающих доноров; interaction-aware вариант против аддитивного; стоимость подготовки и точка окупаемости. Сравнения должны сохранять label budget и число FM-evaluations. Имена/ID donor и скрытые query outcomes не входят в deployment-предиктор.

Сохраняется риск, что causal-reward версия VIP-COP или существующего learned retriever полностью объяснит выигрыш. Связанные CRUMB/VIP/Rank-Learner разбираются в соседних tracks; здесь они не пересказываются. Вывод о методе должен следовать из сравнения фактической policy value, а не из переименования известных блоков.

## Доступ и нерешённые вопросы

Для Hansen получен оригинальный publisher abstract; тело OUP недоступно. У Rosenbaum–Rubin подтверждены publisher metadata и summary оригинальной сканированной статьи; полного text-access нет. У GLISTER издатель блокирует бота, использован авторский arXiv full text с подтверждённым journal reference. Для COT/RetICL/KDSM подтверждены версии препринтов, последующие venue не выдуманы. UPRISE имеет расхождение имени Deng между ACL metadata и PDF; карточка это сохраняет. Exact matching novelty всей комбинации остаётся открытой после этого целевого обзора.

## Дополнительный концептуальный мост, октябрь 2026

[Blackwell framework, C30](https://www.mdpi.com/2504-4990/8/10/336) помогает разделить информативность источника и реализованную полезность конкретной замороженной модели. Его невозможность универсального scalar ranking относится к task-independent score; её нельзя использовать как доказательство против любого query-conditioned selector. Для нашего проекта это мотивация измерять actual FM response, а не новая теорема о превосходстве utility-swap. Direct full-text доступ вернул 429; проверены publisher-indexed методические фрагменты и metadata, поэтому evidence flag консервативный.
