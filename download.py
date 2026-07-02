import studies  # noqa: F401  — registers Pinet2024Meg / Pinet2024Eeg
from neuralset.events import Study

study = Study(name="Pinet2024Meg", path="SpanishBCBL")  # "Pinet2024Eeg" for EEG
study.download()        # fetch this study's recordings + logs from the HF Hub into `path`
events = study.build()  # standardized event dataframe across all subjects/sessions