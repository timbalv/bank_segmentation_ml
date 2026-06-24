# Banki Ügyfélcsoportosítás Mesterséges Intelligencia Alapú Algoritmusokkal

Ez a repozitórium a szakdolgozatom gyakorlati megvalósítását, a forráskódokat és a kapcsolódó adatelemzési kísérleteket tartalmazza. A kutatás célja a bankszektorban alkalmazható gépi tanulási alapú ügyfélszegmentációs eljárások összehasonlítása és üzleti hatékonyságuk értékelése volt.


## 📌 Projekt áttekintés
A projekt során a Moro et al. (2014) által publikált portugál banki marketing adathalmazt elemeztem, amelyet az UCI Machine Learning Repository oldaláról találtam. A kutatás fókuszában két fő megközelítés állt:
1. **Statikus szegmentáció:** Életkor és számlaegyenleg alapján (K-means és DBSCAN).
2. **Viselkedési szegmentáció:** Hívásszám és híváshossz alapján (DBSCAN), kampányoptimalizálási és CRM célokból.


## Fájlstruktúra és a kódok leírása

A repozitóriumban található szkriptek a kutatás különböző fázisait és a modellek evolúcióját mutatják be:

* **`bank+marketing`**: A banki adathalmazt tartalmazó adattáblákat tartalmazó mappa. Benne a * **`bank-full.csv`** táblát alkalmaztam a klaszterezések során.

* **`kmeans_v1.ipynb`**: A véglegesített K-means modell, amely a statikus (életkor-számlaegyenleg) változópároson sikeresen azonosított 3 jól elkülöníthető, üzletileg releváns marketing-szegmenst winsorizált adatokon.

* **`kmeans_v2.ipynb`**: A K-means modell viselkedési változókkal (hívásszám-híváshossz) 3 klasztert eredményezett optimálisan, de ezzel a későbbi DBSCAN modellhez képest gyengébb üzleti interpretálhatóságot ad.
A k-means klaszterezésekhez tartozó export mappa: **`exports_kmeans`**

* **`dbscan_v1.ipynb`**: Az első DBSCAN próbálkozás (paramétertér-kutatás, Eps és MinPts kalibráció a k-dist függvény segítségével). Munkafüzethez tartozó export mappa: **`exports_dbscan_v1`**

* **`dbscan_v2.ipynb`**: Kísérletek a DBSCAN finomhangolására, ez végül rávilágított az algoritmus korlátaira a folytonos, sűrűségbeli elkülönülés nélküli adathalmazokon. Munkafüzethez tartozó export mappa: **`exports_dbscan_v2`**

* **`dbscan_v3.ipynb`**: A véglegesített, sikeres DBSCAN modell, amely a viselkedési változópároson (hívásszám-híváshossz) működve hatékonyan különítette el a tipikus ügyfeleket a zajpontoktól (anomáliáktól), megalapozva egy automatizált CRM kampányoptimalizálási szabályrendszert. Munkafüzethez tartozó export mappa: **`exports_dbscan_v3`**


A végleges változatból kimaradt tartalom:
* **`x-means-v1.ipynb`**: Kísérlet az **X-means klaszterezéssel**. A szakdolgozat tömörítésének és egyértelműsítésének szempontjából ki kellett vennem. Munkafüzethez tartozó export mappa: **`exports_xmeans`**

* **`hierarchical-clustering-v1.ipynb`**: Kísérletek a **Hierarchikus klaszterezéssel**. 2- (92% középosztálybeli többség, 8% gazdag kisebbség) és 3-klaszteres (nagyjából megegyezik a K-means statikussal) modellek számítottak ideálisnak a metrikák alapján. Bár sikeresen lefutott, a szakdolgozat tömörítésének és egyértelműsítésének szempontjából ki kellett vennem. Munkafüzethez tartozó export mappa: **`exports_hacluster`**


## Alkalmazott technológiák és algoritmusok
* **Nyelv:** Python 3.12
* **Főbb könyvtárak:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Algoritmusok:**
    * K-means (Távolságalapú klaszterezés, WCSS, Elbow-módszer, Silhouette score, Davies-Bouldin-index,         Calinski-Harabasz-index)
    * DBSCAN (Sűrűség- és lokális alapú klaszterezés, zajszűrés, Gini együttható)
    * Hierarchikus Agglomeratív Klaszterezés (Dendrogram elemzés)
    * X-means (Bayesian Information Criterion alapú automatikus klaszterszám-választás)


## Főbb Kutatási Eredmények
* A **K-means** kiválóan teljesített a demográfiai és pénzügyi (statikus) adatokon, és sikeresen támogatja a differenciált termékajánlatok (pl. prémium bankolás, megtakarítási számlák) célzását.
* A **DBSCAN** a statikus adatok természetes folytonossága miatt ott alkalmatlannak bizonyult (az adatok mesterséges transzformációját, pl. binninget az üzleti értelmezhetőség megőrzése érdekében elvetettem). Ugyanakkor a **viselkedési adatokon (v3)** kimagasló eredményt hozott: a call center erőforrás-pazarlásának minimalizálására alkalmas CRM szabályt sikerült vele felállítani a felesleges hívások kiszűrésével.


---
*A projekt oktatási és kutatási célból jött létre.*