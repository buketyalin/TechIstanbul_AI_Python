# -*- coding: utf-8 -*-

"""
===============================================================================
MINI MACHINE LEARNING PROJESI - E-POSTA SPAM TAHMINI
===============================================================================

Bu proje, kullanicinin disaridan bir CSV dosyasi secmesini ve bu veri uzerinde
console/terminal araciligiyla temel Machine Learning adimlarini uygulamasini
saglar.

PROJENIN AMACI
--------------
Bir e-postanin SPAM olup olmadigini tahmin eden bir Classification uygulamasi
olusturmaktir.

ORNEK CSV SUTUNLARI
-------------------
kelime_sayisi
link_sayisi
buyuk_harf_orani
supheli_kelime_sayisi
gonderici_puani
ek_var
spam

Ornek:
kelime_sayisi,link_sayisi,buyuk_harf_orani,supheli_kelime_sayisi,gonderici_puani,ek_var,spam
120,0,0.05,0,92,0,0
45,6,0.72,5,18,1,1

spam:
    0 -> Normal e-posta
    1 -> Spam e-posta

ONEMLI
------
Bu proje icin hedef sutun otomatik olarak 'spam' kabul edilir.
'spam' disindaki sutunlar feature olarak kullanilir.

Program sayisal ve kategorik sutunlari otomatik algilar.
"""

# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# Asagidaki import bolumu, projenin ihtiyac duydugu kutuphaneleri programa
# dahil eder.
#
# os / pathlib:
#   Dosya ve klasor islemleri icin kullanilir.
#
# pandas:
#   CSV dosyasini okumak, tablo halinde incelemek ve temizlemek icin kullanilir.
#
# numpy:
#   Sayisal islemlerde ve veri tipleriyle calisirken kullanilir.
#
# matplotlib:
#   Confusion Matrix grafigini PNG olarak kaydetmek icin kullanilir.
#
# scikit-learn:
#   Veriyi train/test olarak ayirmak, on isleme yapmak, model egitmek ve
#   Accuracy, Precision, Recall, F1 gibi metrikleri hesaplamak icin kullanilir.
# -----------------------------------------------------------------------------
from pathlib import Path
from tkinter.ttk import Style
from typing import Optional, List, Dict, Any

# Path, dosya ve klasörlerin yollarını yönetmek için kullanılır.
# tkinter, Python ile grafiksel kullanıcı arayüzü (GUI) oluşturmak için kullanılan standart kütüphanedir.
# Style ise GUI'deki buton, label, entry gibi widget'ların görünümünü/biçimini değiştirmek için kullanılır.
# 'from typing import Optional, List, Dict, Any' : Bunlar Python'daki type hinting (tür belirtme) için kullanılır.
# Optional: Bir tür veya None
# List: Liste türü
# Dict: Dictionary türü
# Any: Herhangi bir tür

# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from colorama import Fore

from sklearn.pipeline import Pipeline


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# AppState sinifi program boyunca kullanilan verileri tek bir yerde tutar.
#
# Neden gereklidir?
# Console uygulamalarinda kullanici once CSV yukler, sonra temizleme yapar,
# sonra model egitir. Her adimda ayni veriyi tekrar tekrar okumak yerine
# programin mevcut durumunu burada sakliyoruz.
#
# raw_df:
#   CSV dosyasindan ilk okunan, dokunulmamis orijinal veri.
#
# df:
#   Temizleme ve analiz islemlerinde kullanilan aktif veri.
#
# target_column:
#   Tahmin edilmek istenen hedef sutun.
#
# feature_columns:
#   Modelin tahmin yaparken kullanacagi giris sutunlari.
#
# best_model:
#   Egitilen modeller arasinda F1 skoruna gore en basarili model.
# -----------------------------------------------------------------------------
class AppState: # AppState adında bir sınıf oluşturuyoruz.
    def __init__(self): # bu sınıftan yeni bir nesne oluşturulduğunda otomatik çalışır.
        # self.xxx : Uygulamanın durumunu tutan değişken
        self.csv_path: Optional[Path] = None # csv dosyasının yolunu tutar.
                                             # None: Henüz CSV seçilmemiş
        self.raw_df: Optional[pd.DataFrame] = None  # csv'den okunan orijinal,
                                                    # değiştirilmemiş DataFrame burada tutulacak
        self.df: Optional[pd.DataFrame] = None # Veri temizlendikten/preprocess edildikten
                                               # sonraki DataFrame burada tutulabilir.

        self.target_column: Optional[str] = None # Modelin tahmin etmeye çalıştığı sütunlar
        self.feature_columns: List[str] = [] # Modelin tahmin yapmak için kullanacağı girdi sütunlar
                                             # başlangıçta liste boş

        self.best_model: Optional[Pipeline] = None # Model karşılaştırıldıktan sonra seçilen modeli
        self.best_model_name: Optional[str] = None # ve ismini tutuyor

        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        self.y_pred: Optional[np.ndarray] = None

        self.model_results: List[Dict[str, Any]] = [] # Birden fazla modelin sonuçlarını saklamak için kullanılır.

        # Program ilk açıldığında yalnızca veri hazırlama adımları (1-6)
        # gösterilir. Veri temizleme başarıyla tamamlandığında ikinci aşama
        # yani Machine Learning seçenekleri açılır.
        self.preprocessing_completed: bool = False # preprocessing tamamlandığında =True olacak

        # Aktif console ekranini takip eder.
        # 1 = Veri Hazirlama, 2 = Machine Learning.
        self.current_step: int = 1 # uygulamanın hangi adımda olduğunu tutar

        # Aktif olarak hangi veriyle devam edildigini takip eder.
        # Degerler: 'original', 'cleaned' veya None
        self.active_data_source: Optional[str] = None # verinin nereden geldiğini tutar. Örn: "GitHub" veya "Local CSV"
        self.cleaned_csv_path: Optional[Path] = None # temizlenmiş CSV'nin yolu
        self.last_pdf_report_path: Optional[Path] = None # en son oluşturulan PDF raporunun dosya yolu



# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# console ekranının daha okunabilir hale gelmesini sağlamak
# SOLID: Single Responsibility
def print_header(title:str) -> None: # ->:Return yapısında bir şey yoksa başlangıcı None olarak al demek
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# Menü kullanıcının sonucu okuyabilmesini için ENTER
def pause() -> None:
    input("\nDevam etmek için lütfen ENTER tuşuna basınız...")


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# Menu yazısını farklı renklerde kullanmamızı sağlar
def print_menu_option(text:str) -> None:
    print(Fore.LIGHTCYAN_EX + text + Style.RESET_ALL) # Style: HTML bir yapının iskeletiyse CSS ise onun makyajıdır.

# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# print_step_title fonskiyon STEP başlıklarını menu seçeneklerinden ayırmak için parlak camgöbeği renkte gösterir.
def print_step_title(text:str) -> None:
    print(Fore.CYAN + Style.BRIGHT+ text + Style.RESET_ALL)


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# CSV sutun adlarını daha düzenli hale getirmek
# Ornek:
# " KeliME Sayısi " -> "kelime_sayisi"
# Bu sayede sutun isimlerindeki boşluk, büyük/küçük harf farklarlarından kaynaklanan hataları azaltmak
def normalize_column_name(name:str) -> str:
    value = str(name).replace("\ufeff","").strip().lower()

    replacements = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        " ": "_",
        "-": "_",
        "/": "_",
        "\\": "_",
    }

    for old, new in replacements.items():
        value = value.replace(old,new)

    while "__" in value:
        value= value.replace("__","_")

    return value.strip("_")



# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# discover_csv_files fonskiyonu kullanicinin dosya seçebilmesini için CSV dosyalarını tarar ve sadece görününe CSV dosyalarını eklemeye yani dinamik olarak csv dosyalarını seçmeye yarar.
def discover_csv_files() -> List[Path]:
    found: List[Path] = []

    search_dirs = [
        Path.cwd(),
        Path.cwd() / "data"
    ]

    for folder in search_dirs:
        if not folder.exists() or not folder.is_dir():
            continue

        for file_path in folder.glob("*.csv"): # * her şey demek. Yani adı herhangi bir şey olabilir ama csv dosyası olsun
            resolved = file_path.resolve()
            if resolved not in found:
                found.append(resolved)
    return sorted(found, key=lambda p: p.name.lower())


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# Manuel olarak (Copy/Paste) olarak girilen yolu seçmek
#  Seçeneklerden
def choose_csv_path() -> Optional[Path]:
    print_header("CSV DOSYASINI SEÇ")

    # -----------------------------------------------------------------
    # BU MENU NE ISE YARAR?
    # -----------------------------------------------------------------
    # Kullanici CSV dosyasini iki farkli yontemle secebilir:
    #
    # 0 - Ana menuye don
    #     CSV secmeden STEP 1 ana menusune geri doner.
    #
    # 1 - Dosya yolunu manuel gir
    #     Kullanici CSV dosyasinin tam yolunu klavyeden yazar.
    #
    #     Ornek:
    #         E:\ML\veriler\spam.csv
    #
    # 2 - Dosya yolunu dosya secerek gir
    #     Windows/Linux dosya secme penceresi acilir.
    #     Kullanici CSV dosyasini tiklayarak secer.
    # -----------------------------------------------------------------
    print_menu_option("0 -Ana menüye dön")
    print_menu_option("1 -Dosya yolunu manuel gir")
    print_menu_option("2 -Dosya yolunu dosya seçerek gir")

    choice = input("\nSeciminiz: ").strip()

    if choice == "0":
        return None

    if choice == "1":
        raw_path = input(
            "\nCSV dosyasını tam yolunu giriniz: "
        ).strip().strip('"')

        if not raw_path:
            print("\nHATA: Dosya yolu boş bırakılamaz.")
            return None
        # expanduser : kısayolları gerçek klasor yoluna çevirmeye yarar.
        # ~\Desktop\veri.csv C:\Users\Data\Desktop\veri.csv
        path = Path(raw_path).expanduser()

        if not path.exists():
            print("\nHATA Girilen dosya bulunamadı.")
            return None

        if not path.is_file():
            print("\nHATA Girilen yol dosya değil.")
            return None

        if  path.suffix.lower() != ".csv":
            print("\nHATA Girilen dosya CSV uzantılı değil.")
            return None

        print(f"\nSeçilen CSV dosyasi:\n{path.resolve()}")
        return path.resolve()

    # -----------------------------------------------------------------
    # BU MENU NE ISE YARAR?
    # -----------------------------------------------------------------
    # tkinter: Python standart kütüphanelerinden biridir.
    # Kullanıcının fare ile dosya seçerek programa aktarılmasıdır.
    if choice == "2":
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()

            # Dosya seçme penceresinin arkada kalmasını engelemeye çalışır.
            try:
                root.attributes("-topmost",True)
            except Exception: # Exception = Program çalışırken ortaya çıkan ve normal akışı bozan bir durum.
                pass          # hata oluşursa program kapatılmaz, devam eder.

            selected_file = filedialog.askopenfilename(
                title="CSV Dosyasını Seç",
                filetypes=[
                    ("CSV Dosyaları", "*.csv"),
                    ("Tüm Dosyalar", "*.*"),
                ]
            )

            root.destroy()

            if not selected_file:
                print("\nDosya seçimi iptal edildi")
                return None

            path = Path(selected_file)

            if not path.exists(): # belirtilen konumda dosya/klasör var mı?
                print("\nHATA: Seçilen dosya bulunamadı")
                return None

            if  path.suffix.lower() != ".csv": # suffix dosyanın uzantısını kontrol ediyor
                print("\nLütfen CSV uzantıli bir dosya seçiniz.")
                return None

            print(f"\nSeçilen CSV dosyasi:\n{path.resolve()}") # path.resolve() dosyanın tam/absolute yolunu verir.
            return path.resolve()
        except ImportError:
            print(
                "\nHATA: Bu python sürümünde tkinter bulunamadı.\n"
                "Alternatif olarak 1- Dosya yolunu manuel gir seçeneğini de kullanabilirsiniz "
            )
            return None
    print("\nHATA: 0<=X<=2 arasında tam sayı seçmelisiniz yani 0,1,2 kullanabilirsiniz")
    return None


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# CSV dosyasını pandas DataFrame formatında okur.
# Separator: Virgül, noktalı virgül vb pandas tarafından otomatik olarak tahmin etmesine yardımcı olan metriklerdir
def read_csv_safely(path:Path) -> pd.DataFrame:
    encodings =[ "utf-8", "utf-8-sig", "latin-1"]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=encoding
            )
        except Exception as exc:
            last_error =exc

    raise RuntimeError(
        f"CSV dosyasi okunamadi. Son hata: {last_error}"
    )


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# load_csv
# CSV dosyasını okuyup AppState içindeki bilgileri güncelliyor.
def load_csv(state: AppState) ->None:
    path = choose_csv_path()

    if path is None:
        return

    try:
        df = read_csv_safely(path)

        if df.empty: # DataFrame'in boş olup olmadığını kontrol eder.
            print("\nHATA: CSV dosyasi boş")
            return

        df.columns = [normalize_column_name(col) for col in df.columns] # Sütun isimlerini standart bir hale getiriyor. Örn: " Hasta Ysşı " -> "hasta_yasi"

        state.csv_path=path
        state.raw_df= df.copy(deep=True) # Pandas'ın gerçekten bağımsız bir kopya oluşturabilmesi için "deep=True" yazılması lazım.
        state.df= df.copy(deep=True)

        state.target_column = None
        state.feature_columns = []

        state.best_model = None
        state.best_model_name = None

        state.X_test = None
        state.y_test = None
        state.y_pred = None

        state.model_results=[]

        state.current_step =2
        state.preprocessing_completed =False
        state.cleaned_csv_path = None
        state.active_data_source ="original"

        print_header("CSV BAŞARIYLA YÜKLENDİ")

        file_size_kb = path.stat().st_size /1024
        missing_total = int(df.isna().sum().sum())
        duplicated_total = int(df.duplicated().sum()) # tekrarlanan satır sayısını bulur

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist() # sayısal veri tipindeki sütunları seçer, sütun isimlerini alır ve listeye dönüştürür.
        categorical_columns=[ # sayısal sütunların içinde olmayanları listeye dönüştürür
            column for column in df.columns
            if column not in numeric_columns
        ]

        target_column = "spam" if "spam" in df.columns else None # eğer sütunda spam yazıyorsa dahil ediyoruz, yazmıyorsa hiçbir şey yapmıyoruz.
        features_columns = [
            column for column in df.columns
            if column != target_column
        ]

        print(f"Dosya Adı:              {path.name}")
        print(f"Dosya Yolu:             {path}")
        print(f"Dosya Boyutu:           {file_size_kb:.2f} KB")
        print(f"Dosya Satır sayısı:     {len(df)}")
        print(f"Dosya Sutun sayısı:     {len(numeric_columns)}")
        print(f"Kategorik Sutun   :     {len(categorical_columns)}")
        print(f"Eksik Değer  :          {missing_total}")
        print(f"Duplicate Satır  :      {duplicated_total}")

        if target_column:
            print(f"Target / Label      : {target_column}")
            print(f"Problem Türü        : classification")
            print(f"Feature Sayısı      : {len(features_columns)}")

        else :
            print(f"Target / Label      :  BULUNAMADI ")
            print(f"Problem Türü      : Belirtilmedi")
            print(f"UYARI     :  CSV içinde 'spam' sutunu bulunamadı")

        print("\nFeature Sutunları")
        for column in features_columns:
            print(f"- {column}")

        if target_column:
            print("\nTarget / Label")
            print(f"- {target_column}")

        print("\nCSV kullanima hazir")
    except Exception as exc:
        print(f"\nHATA: CSV yüklenmedi. \n{exc}")


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# CSV dosyası yüklenmeden menü çalışmasını engelle
def require_data(state: AppState) -> bool:
    if state.df is None:
        print("\nÖnce bir CSV dosyasını yüklemelisiniz.")
        return False
    return True