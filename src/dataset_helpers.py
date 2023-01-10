import datetime
import pandas as pd
import numpy as np
import re
import json
from os import path
from lemmatizer import lemmatize
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from nltk.tokenize import WhitespaceTokenizer
from file_paths import data_path
'''
Use conda env turkunlp_neural_parser to be able to use lemmatizer
'''

def count_tokens(str_input):
    """
    Count tokens from str input with WhitespaceTokenizer
    Arguments:
        str_input (str): string to be tokenized
    Return:
        token_n (int): number of tokens
    """
    tokenizer = WhitespaceTokenizer()
    token_n = len(tokenizer.tokenize(str_input))
    return token_n


def calculate_time_difference(diagnosis,reservation):
    """
    Calculate time difference of two numpy datetime objects

    Arguments:
        begining_time (numpy.datetime64): start time of the event
        end_time (numpy.datetime64): end time of the event
    
    Return:
        difference (int): positive value if diagnosis is before reservation
    """
    difference = reservation-diagnosis
    #time difference in days
    difference = int(difference.astype('timedelta64[h]').astype(np.int32)/24)
    return difference

def add_diagnose_column(data,service_df):
    '''
    Add diagnose column to dataframe

    Arguments:
        data (pandas.DataFrame): df indexed with uniq uids
        diagnose_data (pandas.DataFrame): service df containing all diagnoses given to patient, indexed with uniq uids
    Return:
        None
    '''
    missin_diagnose_n=0
    data['diagnose']=''
    dia_re = re.compile('[E][1][0]\.[0-9]')
    can_re_1 = re.compile("[C][0-8][0-9]\.[0-9]")
    can_re_2 = re.compile("[C][9][0-7]\.[0-9]")
    can_re_3 = re.compile("[D][3][3]\.[0-9]")
    can_re_4 = re.compile("[D][3][5]\.[3][0]")
    can_re_5 = re.compile("[D][4][6]\.[0-9]")
    can_re_6 = re.compile("[D][4][7]\.[0-9]")
    print("Starting calculating diagnostic statistics")
    for u_i in tqdm(data.index.values):
        all_diagnoses = service_df.loc[u_i,'diagnose_code']
        if  isinstance(all_diagnoses,pd.core.series.Series):
            all_diagnoses = all_diagnoses.dropna()
            all_diagnoses=list(all_diagnoses)
        if isinstance(all_diagnoses,pd.DataFrame):
            all_diagnoses = list(all_diagnoses.dropna().to_numpy(dtype=str))
        if isinstance(all_diagnoses,list):
            all_diagnoses = " ".join(all_diagnoses)

        if pd.isna(all_diagnoses):
            print(f'Uid {u_i} do not have any diagnose')
            print(service_df.loc[u_i,'diagnose_code'])
            missin_diagnose_n+=1
            continue

        if re.search(can_re_1,all_diagnoses) or re.search(can_re_2,all_diagnoses) or re.search(can_re_3,all_diagnoses) or re.search(can_re_4,all_diagnoses) or re.search(can_re_5,all_diagnoses) or re.search(can_re_6,all_diagnoses):
            prev_value = data.loc[u_i,'diagnose']
            if 'C' not in prev_value:
                data.loc[u_i,'diagnose']=prev_value+'C'
          
        if re.search(dia_re,all_diagnoses):
            prev_value = data.loc[u_i,'diagnose']
            if 'E' not in prev_value:
                data.loc[u_i,'diagnose']=prev_value+'E'
    no_diagnose = data[data['diagnose']=='']
    indexes = list(no_diagnose.index.values)
    no_diagnose_services = service_df.loc[indexes]
    missin_diagnose_n+=len(indexes)
    no_diagnose_services.to_csv(f"{data_path}/no_diagnose_services.csv")
    data = data[data['diagnose']!='']
    return data, missin_diagnose_n

def create_dataset(services_df,text_df,mental_health_related_codes,diagnosis='C-E',late_effect=365,days_to_subtract=7,lemmatize_text=True,vectorize=True,save_files=False,save_path=data_path,devel_mode=True):
    """
    Create dataset

    Arguments:
        services_df (pandas.DataFrame): df indexed with uids, information about reservations
        text_df (pandas.DataFrame): df indexed with uids, patient texts
        mental_health_related_codes (list): codes related to mental health reservations
        diagnosis (str): diagnosis codes found in patient data, give <E> for diabetes, <C> for cancer and tumour disease and <C-E> for all
        late_effect (int): number of days to be late effect
        days_to_subtract (int): number of days removed before reservation
        save_files (bool): whether to save logs and csv files
        save_path (str): path where to save files
        lemmatize_text (bool): wheter to lemmatize notes
        vectorize_text (bool): wheter to create vector representation
        save_files (bool): wheter to save dataset as csv
        save_path (str): path for output csv
        devel_mode (bool): if True only 10% of data is lemmatized
    Returns
        return_df (pandas.DataFrame): new df
        logs (dict): descriptive statictics about the dataset
        None: if save_files is True, python objects are not returned

    """
    patient_uids = services_df.index.unique()
    p_index = {'patient':patient_uids}
    return_df = pd.DataFrame(data=p_index)
    return_df['texts']=''
    return_df['label']=0
    return_df = return_df.set_index('patient')
    return_df,missing_diagnose = add_diagnose_column(return_df,services_df)
    if diagnosis == 'C':
        return_df = return_df[return_df['diagnose'].str.match('[C]') == True]
    
    missing_text_count = 0
    patient_only_one_r = 0
    only_one_text = 0
    too_many_missing_timestamps=0
    too_many_missing_notes=0
    uids_to_drop = []
    mh_before_d = 0
    n_documents = []
    
    for u_i in return_df.index.values:
        if u_i not in text_df.index:
            print(f"Any patient texts were not found with uid: {u_i}")
            missing_text_count+=1
            uids_to_drop.append(u_i)
            continue
        
        one_patient_services = services_df.loc[u_i]
        one_patient_texts = text_df.loc[u_i]
       
        if isinstance(one_patient_services,pd.core.series.Series):
            print(f"Patient {u_i} has only one reservation")
            patient_only_one_r+=1
            uids_to_drop.append(u_i)
            continue

        if isinstance(one_patient_texts,pd.core.series.Series):
            print(f"Patient {u_i} has only one text")
            only_one_text+=1
            uids_to_drop.append(u_i)
            continue
        
        if one_patient_texts[one_patient_texts['text_timestamp'].isna()].values.any():
            num_of_missing_timestamps = one_patient_texts['text_timestamp'].isna().values.sum()
            if num_of_missing_timestamps>len(one_patient_texts)*0.05:
                print(f"Patient {u_i} has too many missing text timestamps {num_of_missing_timestamps} / {len(one_patient_texts)}")
                too_many_missing_timestamps+=1
                uids_to_drop.append(u_i)
                continue
            else:
                og_len = len(one_patient_texts)
                one_patient_texts = one_patient_texts[one_patient_texts['text_timestamp'].notna()]
                assert og_len>len(one_patient_texts)

        if one_patient_texts[one_patient_texts['texts'].isna()].values.any():
            num_of_missing_notes = one_patient_texts['texts'].isna().values.sum()
            if not num_of_missing_notes<len(one_patient_texts)*0.05:
                too_many_missing_notes+=1
                uids_to_drop.append(u_i)
                continue
            else:
                one_patient_texts = one_patient_texts[one_patient_texts['texts'].notna()]
        
        if len(one_patient_texts)==1:
            only_one_text+=1
            uids_to_drop.append(u_i)
        #find patients who have received care in mental health unit
        mental_health=one_patient_services[one_patient_services['labels'].isin(mental_health_related_codes)]
        #drop ones that has not date
        mental_health=mental_health[~mental_health.label_timestamp.isnull()]
        if mental_health.empty:
            one_patient_texts = one_patient_texts.sort_values(by='text_timestamp')
            return_df.loc[u_i,'texts']=" ".join(one_patient_texts['texts'].tolist())
            n_documents.append(len(one_patient_texts))
            continue

        if isinstance(mental_health,pd.core.series.Series):
            first_m_reservation_time=mental_health['label_timestamp']

        else:
            mental_health = mental_health.sort_values(by="label_timestamp")
            #find first mental health unit reservation
            first_m_reservation_time = mental_health.iat[0, 0]
            first_m_reservation_time = first_m_reservation_time.to_numpy()

        #find first cancer or diabetes diagnosis
        one_patient_diagnosis = return_df.loc[u_i,'diagnose']
        diagnosis_found = False
        if 'C' in one_patient_diagnosis:
            cancer_diagnosis = one_patient_services[(one_patient_services['diagnose_code'].str.match('[C][0-8][0-9]\.[0-9]') == True)|
                                                    (one_patient_services['diagnose_code'].str.match('[C][9][0-7]\.[0-9]') == True)|
                                                    (one_patient_services['diagnose_code'].str.match('[D][3][3]\.[0-9]') == True)|
                                                    (one_patient_services['diagnose_code'].str.match('[D][3][5]\.[3][0]') == True)|
                                                    (one_patient_services['diagnose_code'].str.match('[D][4][6]\.[0-9]') == True)|
                                                    (one_patient_services['diagnose_code'].str.match('[D][4][7]\.[0-9]') == True)
                                                    ]
            diagnosis_found = True

        if 'E' in one_patient_diagnosis and diagnosis!='C' and not diagnosis_found:
            cancer_diagnosis = one_patient_services[one_patient_services['diagnose_code'].str.match('[E][1][0]\.[0-9]') == True]

        cancer_diagnosis = cancer_diagnosis.sort_values(by="diagnose_timestamp")
        first_cancer_diagnosis_time = cancer_diagnosis.iat[0, 3]
        first_cancer_diagnosis_time = first_cancer_diagnosis_time.to_numpy()
        #calculate_time_difference(beginning_time,end_time)
        time_difference=calculate_time_difference(first_cancer_diagnosis_time,first_m_reservation_time)
        if time_difference>late_effect:
            #remove docs one week before the reservation
            d_to_substarct = np.array([datetime.timedelta(days=days_to_subtract)], dtype="timedelta64[ms]")[0]
            d = first_m_reservation_time - d_to_substarct
            one_patient_texts = one_patient_texts[one_patient_texts['text_timestamp']<d]
            one_patient_texts = one_patient_texts.sort_values(by="text_timestamp")
            return_df.loc[u_i,'texts']=" ".join(one_patient_texts['texts'].tolist())
            return_df.loc[u_i,'label']=1
            n_documents.append(len(one_patient_texts))

        else:
            d_to_substarct = np.array([datetime.timedelta(days=days_to_subtract)], dtype="timedelta64[ms]")[0]
            d = first_m_reservation_time - d_to_substarct
            one_patient_texts = one_patient_texts[one_patient_texts['text_timestamp']<d]
            if one_patient_texts.empty:
                print(f"patient {u_i} have had MH-related contact before diabetes or cancer diagnosis")
                uids_to_drop.append(u_i)
                mh_before_d+=1
                continue
            else:
                one_patient_texts = one_patient_texts.sort_values(by="text_timestamp")
                return_df.loc[u_i,'texts']=" ".join(one_patient_texts['texts'].tolist())
                n_documents.append(len(one_patient_texts))

    return_df = return_df.drop(uids_to_drop,axis=0)

    if lemmatize_text:
        if devel_mode:
            return_df = return_df.sample(frac=0.01)
        if path.exists(f"{data_path}/lemmatized_notes.csv"):
            lemmatized_df = pd.read_csv(f"{data_path}/lemmatized_notes.csv",header=0,index_col=0)
            lemmatized_df = lemmatized_df.loc[list(return_df.index.values)]
            return_df = return_df.join(lemmatized_df,how='left')
            return_df = return_df.drop(columns=['texts'])
            return_df = return_df.rename(columns={"lemmatized_text":'texts'})
        else:
            return_df['texts'] = return_df['texts'].map(lemmatize)
            return_df['token_n'] = return_df['texts'].map(count_tokens)
            return_df.to_csv(f"{data_path}/lemmatized_notes.csv",columns=['texts','token_n'],header=['lemmatized_text','token_n'])
    if vectorize:
        #create vectorizer that uses uni and bigrams, discard words appearing only once
        #max_df = ignore terms that have a document frequency strictly higher than the given threshold
        #min_df =  ignore terms that have a document frequency strictly lower than the given threshold
        if devel_mode:
            countvectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2),max_df=0.9, max_features=2000, dtype=np.int16)
        else:
            countvectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2),max_df=0.8,min_df=5, max_features=2000, dtype=np.int16)  
        #learn the vocabulary dictionary, returns array of shape (n_samples, n_features)
        matrix = countvectorizer.fit_transform(return_df.texts.values)  
        matrix = matrix.toarray()
        # get column names
        columns = countvectorizer.get_feature_names()  
        df_count = pd.DataFrame(matrix,index=return_df.index.values, columns=columns)
        df_count['label']=return_df.label.values
        df_count['diagnose']=return_df.diagnose.values
    #count diagnose statistics
    diagnose_n = return_df.groupby(['diagnose']).count().to_dict()
    diagnose_n = diagnose_n['texts']
    diagnose_n_l = return_df.groupby(['label','diagnose']).size().to_dict()
    diagnose_n_l = dict((f"{k[1]}:{k[0]}", v) for k,v in diagnose_n_l.items())    
    counts=return_df.groupby(['label']).count().to_dict()
    counts = counts['texts']
    mean_n_docs = int(np.mean(n_documents))
    median_n_docs = int(np.median(n_documents))
    max_n_docs = int(np.max(n_documents))
    min_n_docs = int(np.min(n_documents))
    now = datetime.datetime.now()
    logs = {"date_of_creation":now.strftime('%d/%m/%Y %H:%M:%S'),
                "patients_with_only_one_reservation":patient_only_one_r,
                "no_text":missing_text_count,
                "only_one_text":only_one_text,
                "too_many_missing_timestamps":too_many_missing_timestamps,
                "too_many_missing_notes":too_many_missing_notes,
                "mh_contact_before_diagnosis": mh_before_d,
                "n_ehrs":sum(n_documents),
                "n_patients":len(return_df),
                "label_counts":counts,
                'diagnose_counts':diagnose_n,
                'diagnose_counts_by_label':diagnose_n_l,
                'missing_diagnose':missing_diagnose,
                'mean_n_docs':mean_n_docs,
                'median_n_docs':median_n_docs,
                'max_n_docs':max_n_docs,
                'min_n_docs':min_n_docs
                }
    if lemmatize_text:
        logs['token_statistics'] = return_df['token_n'].describe().to_dict()

    if save_files:
        json_object = json.dumps(logs, indent=4)
        with open(f"{save_path}/cancer_data_lemmatized_{lemmatize_text}_vectorized_{vectorize}_diagnosis_{diagnosis}_late_effect_{late_effect}_devel_mode_{devel_mode}.log", "w") as outfile:
            outfile.write(json_object)
        if vectorize:
            df_count.to_csv(f"{save_path}/cancer_data_lemmatized_{lemmatize_text}_vectorized_{vectorize}_diagnosis_{diagnosis}_late_effect_{late_effect}_devel_mode_{devel_mode}.csv")
        else:
            return_df.to_csv(f"{save_path}/cancer_data_lemmatized_{lemmatize_text}_vectorized_{vectorize}_diagnosis_{diagnosis}_late_effect_{late_effect}_devel_mode_{devel_mode}.csv")
        return None
    else:
        if vectorize:
            return  df_count, logs
        else:
            return return_df,logs


