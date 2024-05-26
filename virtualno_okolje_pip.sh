#/bin/bash


#Ta skripta ustvari virtualno okolje in namesti paket pip
#Virtualno okolje je v mapi .venv

VENV=".venv"
#ustvari virtualno okolje
python3 -m venv $VENV

#Aktiviraj virtualno okolje
source $(pwd)/$VENV/bin/activate

#Namesti paket pip  
pip install --upgrade pip
pip install -r requirements.txt

#Deaktiviraj virtualno okolje
# deactivate

#Pobriši virtualno okolje
# rm -rf $VENV






