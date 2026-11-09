fm\_benchmark.py — один запуск кросс-валидации для выбранной модели, датасета, фолда, seed, размера и стратегии контекста.
fm\_holdout.py — проверка зафиксированных конфигураций на независимом holdout.
bootstrap\_criteo\_dopfn.py — парный кластерный бутстрэп главного сравнения Do-PFN на Criteo.
build\_smiles\_report.py — сборка итоговых таблиц, графиков и отчёта из готовых результатов.

Всего исходный эксперимент включал 2700 основных CV-запусков и 48 holdout-запусков.
Проверялись CausalPFN, TabPFN-T и Do-PFN на пяти датасетах.
Стратегии: random, balanced\_treatment, balanced\_treatment\_outcome.
Размеры контекста: 512, 1024, 2048 и 4096.
Seed моделей: 42, 43 и 44. Для CV использовалось пять фолдов.
Метрики: normalized Qini AUC, AUUC и uplift@5%, @10%, @20%, @30%.

Запуск и вспомогательные файлы

build\_manifest.py — подготовка списка CV-заданий.
fm\_array.sbatch и run\_fm\_cv.sbatch — запуск CV через Slurm.
build\_holdout\_plan.py — фиксация выбранных по CV конфигураций.
fm\_holdout\_array.sbatch — запуск holdout через Slurm.
fm\_preflight.sbatch и gpu\_smoke.py — предварительные проверки.
cpu\_smoke.py и check\_project.py — локальные проверки.
fm\_dense\_cache.py и fm\_group\_cache.py — чтение проверенных кешей.
make\_dense\_cache.py, make\_group\_split\_cache.py и make\_dense\_cache.sbatch — создание кешей.
test\_dense\_cache.py и test\_group\_cache.py — тесты кешей.
enable\_dense\_cache.py и enable\_group\_cache.py — исторические скрипты применения патчей. В предоставленной версии fm\_benchmark.py соответствующие изменения уже присутствуют. Повторно применять патчи не требуется.

Условия воспроизведения

В архив не входят датасеты, кеши, сохранённые результаты, веса моделей, отдельный репозиторий Do-PFN и Python-окружения.
На сервере использовались \~/projects/smiles/data/data\_A\_cleaned, \~/projects/Do-PFN и отдельные окружения ml-env и dopfn-gpu-env.
Перед запуском на другом сервере нужно восстановить зависимости и веса моделей, подготовить данные и кеши, проверить настройки и пути в скриптах.
В Slurm-скриптах сохранены исходные абсолютные пути для журналов под /home/sedunovaan. Для другой учётной записи их необходимо заменить до отправки заданий.
В run\_fm\_cv.sbatch значение EVAL\_MAX по умолчанию равно 128 для проверки; основной CV-прогон использовал значения из манифеста, в частности до 8000 оцениваемых строк на фолд.
В fm\_holdout\_array.sbatch по умолчанию указан режим подвыборки; для полного holdout требуется HOLDOUT\_MODE=full и корректный зафиксированный манифест с контрольной суммой.
Не нужно повторно формировать план holdout после просмотра тестовых результатов.

Проверка без запуска моделей

Из папки со скриптами:
python fm\_benchmark.py --inspect-data --data-root /путь/к/data\_A\_cleaned
python test\_group\_cache.py
python test\_dense\_cache.py

