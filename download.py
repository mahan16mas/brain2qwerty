from neuralset.events import Study

study = Study(name="Pinet2024Meg", path="SpanishBCBL")
study.download()
events = study.build()
