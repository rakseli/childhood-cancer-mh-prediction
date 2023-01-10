import pandas as pd
import sys
from dataset_helpers import create_dataset
from file_paths import texts_path,services_path,label_path,data_path
text_df = pd.read_csv(texts_path,header=0,usecols=[1,6,7,8],names=['patient','unit','texts','text_timestamp'],index_col=0)
text_df['text_timestamp'] =pd.to_datetime(text_df['text_timestamp'],format="%Y-%m-%d")
services_df = pd.read_csv(services_path,delimiter=";"
                          ,doublequote=False,escapechar='\'',header=0,usecols=[0,6,16,34,41],
                          names=['patient','label_timestamp','labels','diagnose_code','diagnose_timestamp'],
                         index_col=0)
print(f"Number of patients in text_df: {len(text_df.index.unique())}")
print(f"Number of patients in services_df: {len(services_df.index.unique())}")

print(f"Number of EHRs: {len(text_df)}")
#sys.exit()
services_df['label_timestamp'] = pd.to_datetime(services_df['label_timestamp'].str[:10],format="%Y-%m-%d")
services_df['diagnose_timestamp'] = pd.to_datetime(services_df['diagnose_timestamp'].str[:10],format="%Y-%m-%d")
label_infos = pd.read_csv(label_path,header=0,usecols=[0,2],names=['code','connected'],index_col=0)
label_infos = label_infos[label_infos.connected !=0]
mental_health_related_codes = list(label_infos.index.values)
#remove hematologia
mental_health_related_codes.remove('40H')
#remove lasten ja nuorten sairaanhoito
mental_health_related_codes.remove('40Y')
#remove lastentaudit
mental_health_related_codes.remove('40')
#remove luutuumorien hoito
mental_health_related_codes.remove('20I2')
#remove syöpäklinikka
mental_health_related_codes.remove('65Y')
#remove anestesia ja tehohoito
mental_health_related_codes.remove('11')
#remove kipuklinikka
mental_health_related_codes.remove('11E')

#return_df_single,logs_single=create_dataset(services_df=services_df,text_df=text_df,mental_health_related_codes=mental_health_related_codes,diagnosis='C-E',save_files=False,lemmatize_text=True,vectorize=True,devel_mode=True,save_path=data_path)
#return_df_multi,logs_multi=create_dataset(services_df=services_df,text_df=text_df,mental_health_related_codes=mental_health_related_codes,diagnosis='C',save_files=False,lemmatize_text=True,vectorize=True,devel_mode=True,save_path=data_path)
create_dataset(services_df=services_df,text_df=text_df,mental_health_related_codes=mental_health_related_codes,diagnosis='C-E',save_files=True,lemmatize_text=True,vectorize=True,devel_mode=False,save_path=data_path)
create_dataset(services_df=services_df,text_df=text_df,mental_health_related_codes=mental_health_related_codes,diagnosis='C',save_files=True,lemmatize_text=True,vectorize=True,devel_mode=False,save_path=data_path)

#cancer_diabetes_multi = return_df_multi[return_df_multi['diagnose']=='CE']
#cancer_diabetes_single = return_df_single[return_df_single['diagnose']=='CE']
#cancer_diabetes_multi = cancer_diabetes_multi.loc[list(cancer_diabetes_single.index.values)]
#cb_m_labels=cancer_diabetes_multi.label.values
#cb_s_labels=cancer_diabetes_single.label.values
#assert (cb_m_labels == cb_s_labels).all()
#print("Logs of single",logs_single)
#print("Logs of multi",logs_multi)