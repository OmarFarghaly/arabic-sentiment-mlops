from itertools import cycle

from locust import HttpUser, constant, task


TEXTS = (
    "الخدمة ممتازة.",
    "المنتج سيئ ولا أنصح بشرائه.",
    "التوصيل كان سريعاً والتغليف جيد، وسأطلب مرة أخرى.",
    "السعر مناسب لكن خدمة العملاء لم تساعدني عندما واجهت مشكلة.",
    (
        "طلبت هذا المنتج بعد قراءة عدة مراجعات، ووصل في الموعد المحدد. "
        "كانت العبوة سليمة والتعليمات واضحة، وبعد استخدامه لمدة أسبوع "
        "وجدت أن الجودة جيدة وأنه يلبي احتياجاتي اليومية."
    ),
    (
        "للأسف كانت تجربتي مع المتجر مخيبة للآمال. تأخر الطلب عدة أيام "
        "ووصل أحد المنتجات تالفاً، وحاولت التواصل مع خدمة العملاء أكثر "
        "من مرة دون الحصول على حل واضح أو موعد لاسترداد المبلغ."
    ),
    (
        "اشتريت الجهاز للعمل والدراسة بعد مقارنة عدة خيارات متاحة. "
        "في البداية أعجبني التصميم وخفة الوزن وسهولة الإعداد، "
        "كما كانت الشاشة واضحة ولوحة المفاتيح مريحة للاستخدام الطويل. "
        "لكن البطارية لا تدوم كما توقعت، وعند تشغيل عدة برامج يصبح "
        "الجهاز بطيئاً ويصدر صوتاً مزعجاً. تواصلت مع الدعم وقدموا بعض "
        "الاقتراحات التي حسنت الأداء قليلاً، إلا أن المشكلة لم تختف "
        "تماماً. بشكل عام توجد جوانب جيدة، لكنني لست راضياً عن السعر "
        "مقارنة بالأداء الفعلي."
    ),
)


class ArabicSentimentUser(HttpUser):
    wait_time = constant(1)

    def on_start(self):
        self.texts = cycle(TEXTS)

    @task
    def predict(self):
        with self.client.post(
            "/predict",
            json={"text": next(self.texts)},
            name="/predict",
            timeout=30,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            try:
                data = response.json()
            except ValueError:
                response.failure("Response is not valid JSON")
                return

            if not isinstance(data, dict):
                response.failure("Response must be a JSON object")
                return

            confidence = data.get("confidence")
            if (
                data.get("label") not in {"Negative", "Positive"}
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
            ):
                response.failure("Invalid prediction response")