# Обращение в поддержку Yandex Cloud: DataSphere, нет V100/A100

Где: консоль → Поддержка → Создать обращение → сервис DataSphere, технический вопрос.
Перед отправкой подставьте зону сообщества (страница сообщества в DataSphere).

Сообщество DataSphere bt1t5m46lhju7eabr0qh (зона: <zoneId>), проект bt1qjejdckm5jggapl1k. Задания на g1.1, g1.2, g1.4 (V100) и g2.1, g2.2 (A100) ни разу не получили ВМ: статус ERROR 'Unable to allocate any VM', операция завершается кодом 13 'Internal error' через 80–110 с. Задания (UTC): bt1hr5vc14rvngsd16h4 08.11 07:56 g1.1;
bt1og1n22d7rcd0kebff 08.11 07:58 g1.1;
bt149b6ctkup5l8mdfih 08.11 11:27 g1.1 (операция bt127267442h2ca3mo9f);
bt1erglgpbkt8hguiogq 09.11 17:35 g1.1;
bt1ir54sva4ddo46n5cr g1.2, bt1fk4ouig4tb13ks84s g1.4, bt1ckquen6at1efuhafv g2.1, bt1amj94pl9dptnmffbo g2.2 — 09.11 19:46. x-server-trace-id: c9d93f2431127c0d:75b9f6051c9e18dd:b178ef2051eaeade:1. При этом gt4.1 и gt4i.1 выделяются за ~1 мин (7 ВМ gt4.1 одновременно), c1.4 работает. С 09.11 19:46 UTC создание задания с g1.1 отклоняется: INVALID_ARGUMENT 'Instance types g1.1 are not available for your community'. Кроме того, задания на gt4.1 отклоняются с 'Job execution quota exhausted' (08.11 12:22:28 и 12:47:54 UTC при 0 активных заданий в проекте;
09.11 17:46–17:58 UTC при 8 активных). Вопросы: (1) Доступны ли сейчас ВМ g1.x/g2.x для DataSphere в зоне нашего сообщества после октябрьского инцидента? Если нет — когда ожидаются и в какой зоне есть? (2) Получит ли их новое сообщество в другой зоне на том же платежном аккаунте? (3) Почему g1.1 стал 'not available for your community' и как его вернуть? (4) Каковы значение и область действия квоты на активные задания и можно ли увеличить её до 20?
