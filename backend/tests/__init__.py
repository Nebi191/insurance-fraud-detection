"""Backend test paketi.

Bu `__init__.py` boş görünüyor ama İŞLEVSELDİR, silinmemeli.

pytest'in varsayılan (`prepend`) import modunda, bir test dosyasının paket
kökü `__init__.py` zinciri yukarı takip edilerek bulunur ve o kökün ÜST dizini
`sys.path`'e eklenir. `backend/tests/__init__.py` var olduğu için `backend/`
dizini `sys.path`'e girer ve testler `from app.main import app` diyebilir.

Alternatifi `conftest.py` içinde elle `sys.path` oynamaktı; bu yol daha az
sihirli ve pytest'in kendi kuralına dayanıyor.
"""
