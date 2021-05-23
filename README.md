# reverse_pendulum

Neurális hálózatok c. tárgy házi feladata.

## Környezet

A használt Python környezet beállításai.

### Python verzió

Lokálisan Python 3.8, patch >= 5.

### Python modulok

Python modulok kezeléséhez virtualenv, az alábbi link jó tutorial: <\br>
https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/ </br>

A virtuális környezet neve 'env'. A 'python_requirements.txt' fájlban megtalálhatók a szükséges függőségek.</br>

A használni kívánt Python verzióval kell futtatni a virtualenvet és akkor arra a verzióra készít linkeket.

### Telepítés

TL;DR: install PS szkript futtatása

Manuálisan felrakni a virtualenvet és készíteni egy 'env' nevű környezetet. Ott telepíteni a függőségeket, majd az aktiválás
után a \src mappában a train.py és/vagy eval.py szkriptet kell futtatni.

### Mentés és betöltés

A konfigurációk mentése automatikusan történik a tanítás befejeztével. Minden konfiguráció a saves könyvtáron belülre egy tömörített állományba kerül. A konfiguráció (egyébként beszédes) neve a tanítás alatt többször is látható a parancsorban. </br>
A betöltés és futtatás (még) nem megoldott parancssorból, ehhez az src/eval.py szkriptben szükséges a 'cfgName' változót frissíteni a betölteni kívánt konfiguráció nevére, majd a szkriptet futtatni.