# Итоги исследования на момент остановки, 9 ноября 2026

Работа остановлена по прямому поручению пользователя. Исследовательская цель имеет статус **PAUSED**, не COMPLETE: нового метода выбора контекста с защищаемой новизной и воспроизводимым выигрышем пока нет. Новые эксперименты и исправления последнего пилота не выполняются.

В проекте DataSphere проверены все страницы списка jobs: 38 записей, **0 нетерминальных задач**. Дополнительно обнаруженная выполнявшаяся `whydra-temporal-full` (`bt1r2cjr23mgo8jps41q`) получила принудительную отмену и завершилась **CANCELLED**, 2026-11-08 21:14:38 UTC. Последний H26 (`bt1m3vbfgl0m55k0igq4`) уже завершился **ERROR**, 21:11:02 UTC; API подтвердил, что finished job отменять нельзя. Docker/Podman/containerd процессов и служб локально не обнаружено; единственная WSL Ubuntu имеет статус Stopped. Образы, тома, исходники и научные артефакты не удалялись. Машиночитаемое подтверждение: `repro/hypothesis_search/SHUTDOWN_2026_11_09.json`.

Собран тематический литобзор: 122 научные работы и один программный ресурс; уровень доступа/прочтения отмечен отдельно (93 полных текста, 30 ограниченно доступных источников). Обзор не является PRISMA-переписью. Основной файл — `repro/literature_review_2026-11-08/REVIEW_RU.md`, рядом каталог, карточки и BibTeX. Главный литературный вывод: retrieval, balancing, coreset, model selection, learned context utility, causal policy loss и probabilistic ranking имеют сильных предшественников; простая комбинация известных блоков не устанавливает новый вклад.

В реестре H1–H26 представлены как реальные экспериментальные проверки, так и идеи, остановленные до модели по новизне, математике или materiality. Это **не 26 полных efficacy trials**. Полная история и ссылки — `repro/hypothesis_search/SEARCH_LOG_RU.md`.

| Направление | Подтверждённый результат | Решение |
|---|---|---|
| H1/H2: локальный баланс и weak-effect witnesses | H1 FM MSE +13.8%, хотя direct-local улучшился; H2 FM MSE +1.9% | Эти операторы закрыты |
| H4/H6/H9: fold quotas, coupled sampling, pairwise contexts | H4 не прошёл joint-quality gates; H6 ranking regret +1.31%; H9 regret +32.9% | Закрыты |
| H10/H17: обучаемый critic и pseudoRCT audit | H10 risk prediction не переносится; H17 rank correlation около0.01 и проигрыш mean6 | Не дали полезного критерия выбора |
| H12/H15: actual-FM raw-Y Jacobian criteria | На fresh canonical IHDP20 MSE +46.71% и +43.47% к feature retrieval | Оба метода отвергнуты |
| H19: T/Y quotas для DoPFN | На confounded-null corrected MSE 0.0145737 против0.00330991 контроля, то есть4.40×; есть частичный HTE-policy gain | Общие gates провалены; частичный плюс не спасает метод |
| H20: treatment-coding consistency selector | CATE NMSE0.484814 против0.340402 feature (+42.4%); wins2/20 | Самосогласованность не стала полезным selector |
| H21/H24/H25 | Дубликаты, известная residualization/prognostic matching либо VIP-COP+causal validation; связи с меньшим FM risk не установлено | Не запущены как новые основные методы |
| H22: stratification whole-context ensemble | MC variance всего−3.97% при заранее требуемых20% | Materiality NO-GO |
| H23: CUR approximation context-response panel | Exact conditional MC variance3.324×SRS6; end-to-end на9.4% медленнее six-context batch | Technical materiality NO-GO; CATE quality здесь не оценивалась |

Числа NMSE из разных старых отчётов нельзя сравнивать напрямую: H12/H15 нормировали на Var(trainY), H20 — на Var(queryτ). Проценты в таблице относятся к контролю внутри соответствующего screen. Все последние модельные fits/predictions/gradients выполнялись в DataSphere; локально — подготовка, context routing, проверка исходников и арифметика сохранённых результатов.

Основной эмпирический вывод ограничен проверенными постановками: улучшение геометрии, устойчивости или внутренней согласованности контекста не обеспечило улучшения причинного эффекта. Простые ensembles и native retrieval оказались сильными контролями. Известные orbit averaging/ensembling иногда давали плюс; они не были переименованы в новый метод. Меньшее число query-context evaluations также не гарантирует меньшего времени при native batching.

Последняя H26 дала более конкретный механизм. В pinned CausalPFN двухэтапный10-SD clipping может почти или полностью стереть бинарный X, если minority count ниже

`r(K) = ceil[K(K−1)/(101K−1)]`.

Пороги:2дляK128,6дляK512,11дляK1024. В DataSphere source-only job `bt1l0sjo3upnjedim4sc` независимо проверены168случаев: среди21float32case ниже порога10exact zero и11tiny positive separations≤1e−6; все42float32cases на/выше порога различимы. Это проверка preprocessing, **не FM efficacy**. Общая sigma-clipping алгебра не новая; возможный узкий вклад — диагностика конкретного obstruction и practically useful real-row repair. Exact-real twin error bound требует одного и того же контекста и не переносится автоматически на разные query-specific contexts. DoPFN использует другую мягкую компрессию; общего theorem для всех FM нет.

X-only audit на IHDP не выявил query-exposed событий для feature baseline (0/1500). По raw/category metadata ACIC потенциальная проблема заметнее:42/961feature queries и849/11532shared-context/query pairs. Это exposure indicators, не causal relevance или измеренные ошибки предсказаний. На старых scenario1contexts local greedy с Hamming distances нашёл41/42feasible repairs, медиана3замены;1case имеет недостаточный whole-bank support. Эти данные не доказывают качество финального Euclidean-deployment selector.

H26 efficacy protocol включал12методов и38prospective gates: новый repair, ordinary presence, matched placebo, binary bypass/one-pass/all-off, full-bank, shared random ensemble, native effect-metric retrieval, direct weak learner и binary-scaled kNN. Контексты должны были строиться отдельно по actual T каждого из10ACIC scenarios. Все планы/actors/timings фиксируются до истинных эффектов; joint-win rule требует одновременно обойти raw и оба binary fixes в≥6из8heterogeneous scenarios. Сохранены immutable packages, source reviews и независимый analyzer.

**Эффективность H26 осталась непроверенной.** Последний GPU pilot завершился до FM-предсказаний: я закрепил ошибочное ожидание `Globally binary32 feature contract`, которое фактический preprocessing не подтвердил. В `screen_result.json` зафиксированы `quality_loaded=false` и ошибка до model construction. Это технический INCOMPLETE, не scientific rejection и не успех метода. По просьбе пользователя исправление и перезапуск не выполняются. Frozen protocol: `repro/hypothesis_search/h26_design/efficacy/PROTOCOL_V2_RU.md`; фактический error record: `repro/hypothesis_search/h26_eff_gpu_v2/results/unpacked/results/screen_result.json`.

Для методической short paper сейчас отсутствует необходимый положительный результат. Содержательное основание для диагностической/negative-results работы есть: систематический разрыв между привлекательными context proxies и uplift quality, сильные cost-aware controls, воспроизводимые numerical pitfalls и отдельный H26 preprocessing mechanism. Полноценная публикационная заявка пока не подтверждена: IHDP многократно просмотрена и имеет общих subjects, ACIC scenarios разделяют X, ряд синтетических regimes используют одни банки. Нельзя выдавать множество queries/replications за независимые datasets или заявлять общий impossibility result для выбора контекста.

Все результаты сохранены. Исследование остановлено без завершения исходной научной цели; автоматическое продолжение не запланировано.
