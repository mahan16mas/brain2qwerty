import studies  # ثبت ساختار داده‌ها
from neuralset.events import Study

# اگر داده‌های MEG را می‌خواهید "Pinet2024Meg" و اگر EEG را می‌خواهید "Pinet2024Eeg" بگذرانید
study = Study(name="Pinet2024Meg", path="SpanishBCBL")
study.download()        # دانلود از هگینگ فیس
events = study.build()  # ساخت ساختار رویدادها
print("دانلود و آماده‌سازی با موفقیت انجام شد!")