import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

csv_dir = './csv/'
data_dir = './data/'
def process_admission():
    # Load icustays table
    # Table purpose: Defines each ICUSTAY_ID in the database, i.e. defines a single ICU stay
    print('Load ICU stays...')
    dtype = {'SUBJECT_ID': 'int32',
             'HADM_ID': 'int32',
             'ICUSTAY_ID': 'int32',
             'INTIME': 'str',
             'OUTTIME': 'str',
             'LOS': 'float32'}
    parse_dates = ['INTIME', 'OUTTIME']
    icustays = pd.read_csv(csv_dir + 'ICUSTAYS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)

    print('-----------------------------------------')

    # Load patients table
    # Table purpose: Contains all charted data for all patients.
    print('Load patients...')
    dtype = {'SUBJECT_ID': 'int32',
             'GENDER': 'str',
             'DOB': 'str',
             'DOD': 'str'}
    parse_dates = ['DOB', 'DOD']
    patients = pd.read_csv(csv_dir + 'PATIENTS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)

    # Adjust shifted DOBs for older patients (median imputation)
    old_patient = patients['DOB'].dt.year < 2000
    date_offset = pd.DateOffset(years=(300-91), days=(-0.4*365))
    patients['DOB'][old_patient] = patients['DOB'][old_patient].apply(lambda x: x + date_offset)

    # Replace GENDER by dummy binary column
    patients = pd.get_dummies(patients, columns = ['GENDER'], drop_first=True)

    print('-----------------------------------------')
    print('Load admissions...')
    # Load admissions table
    # Table purpose: Define a patients hospital admission, HADM_ID.
    dtype = {'SUBJECT_ID': 'int32',
             'HADM_ID': 'int32',
             'ADMISSION_LOCATION': 'str',
             'INSURANCE': 'str',
             'MARITAL_STATUS': 'str',
             'ETHNICITY': 'str',
             'ADMITTIME': 'str',
             'ADMISSION_TYPE': 'str'}
    parse_dates = ['ADMITTIME']
    admissions = pd.read_csv(csv_dir + 'ADMISSIONS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)

    print('-----------------------------------------')
    print('Load services...')
    # Load services table
    # Table purpose: Lists services that a patient was admitted/transferred under.
    dtype = {'SUBJECT_ID': 'int32',
             'HADM_ID': 'int32',
             'TRANSFERTIME': 'str',
             'CURR_SERVICE': 'str'}
    parse_dates = ['TRANSFERTIME']
    services = pd.read_csv(csv_dir + 'SERVICES.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)

    print('-----------------------------------------')

    # Link icustays and patients tables
    print('Link icustays and patients tables...')
    icu_pat = pd.merge(icustays, patients, how='inner', on='SUBJECT_ID')
    icu_pat.sort_values(by=['SUBJECT_ID', 'OUTTIME'], ascending=[True, False], inplace=True)

    # Exclude icu stays during which patient died
    icu_pat = icu_pat[~(icu_pat['DOD'] <= icu_pat['OUTTIME'])]

    # Determine number of icu discharges in the last 365 days
    print('Compute number of recent admissions...')
    icu_pat['NUM_RECENT_ADMISSIONS'] = 0
    for name, group in tqdm(icu_pat.groupby(['SUBJECT_ID'])):
        for index, row in group.iterrows():
            days_diff = (row['OUTTIME']-group['OUTTIME']).dt.days
            icu_pat.at[index, 'NUM_RECENT_ADMISSIONS'] = len(group[(days_diff > 0) & (days_diff <=365)])

    # Create age variable and exclude patients < 18 y.o.
    icu_pat['AGE'] = (icu_pat['OUTTIME'] - icu_pat['DOB']).dt.days/365.
    icu_pat = icu_pat[icu_pat['AGE'] >= 18]

    # Time to next admission (discharge to admission!)
    icu_pat['DAYS_TO_NEXT'] = (icu_pat.groupby(['SUBJECT_ID']).shift(1)['INTIME'] - icu_pat['OUTTIME']).dt.days

    # Add early readmission flag (less than 30 days after discharge)
    icu_pat['POSITIVE'] = (icu_pat['DAYS_TO_NEXT'] <= 30)

    # Add early death flag (less than 30 days after discharge)
    early_death = ((icu_pat['DOD'] - icu_pat['OUTTIME']).dt.days <= 30)

    # Censor negative patients who died within less than 30 days after discharge (no chance of readmission)
    icu_pat = icu_pat[icu_pat['POSITIVE'] | ~early_death]

    # Clean up
    icu_pat.drop(columns=['DOB', 'DOD', 'DAYS_TO_NEXT'], inplace=True)

    print('-----------------------------------------')

    # Link icu_pat and admissions tables
    print('Link icu_pat and admissions tables...')
    icu_pat_admit = pd.merge(icu_pat, admissions, how='left', on=['SUBJECT_ID', 'HADM_ID'])

    print('Some data cleaning on admissions...')
    icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('WHITE'), 'ETHNICITY']    = 'WHITE'
    icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('BLACK'), 'ETHNICITY']    = 'BLACK/AFRICAN AMERICAN'
    icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('ASIAN'), 'ETHNICITY']    = 'ASIAN'
    icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('HISPANIC'), 'ETHNICITY'] = 'HISPANIC/LATINO'
    icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('DECLINED'), 'ETHNICITY'] = 'OTHER/UNKNOWN'
    icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('MULTI'), 'ETHNICITY']    = 'OTHER/UNKNOWN'
    icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('UNKNOWN'), 'ETHNICITY']  = 'OTHER/UNKNOWN'
    icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('OTHER'), 'ETHNICITY']  = 'OTHER/UNKNOWN'

    icu_pat_admit['MARITAL_STATUS'].fillna('UNKNOWN', inplace=True)
    icu_pat_admit.loc[icu_pat_admit['MARITAL_STATUS'].str.contains('MARRIED'), 'MARITAL_STATUS']      = 'MARRIED/LIFE PARTNER'
    icu_pat_admit.loc[icu_pat_admit['MARITAL_STATUS'].str.contains('LIFE PARTNER'), 'MARITAL_STATUS'] = 'MARRIED/LIFE PARTNER'
    icu_pat_admit.loc[icu_pat_admit['MARITAL_STATUS'].str.contains('WIDOWED'), 'MARITAL_STATUS']      = 'WIDOWED/DIVORCED/SEPARATED'
    icu_pat_admit.loc[icu_pat_admit['MARITAL_STATUS'].str.contains('DIVORCED'), 'MARITAL_STATUS']     = 'WIDOWED/DIVORCED/SEPARATED'
    icu_pat_admit.loc[icu_pat_admit['MARITAL_STATUS'].str.contains('SEPARATED'), 'MARITAL_STATUS']    = 'WIDOWED/DIVORCED/SEPARATED'
    icu_pat_admit.loc[icu_pat_admit['MARITAL_STATUS'].str.contains('UNKNOWN'), 'MARITAL_STATUS']      = 'OTHER/UNKNOWN'

    columns_to_mask = ['ADMISSION_LOCATION',
                       'INSURANCE',
                       'MARITAL_STATUS',
                       'ETHNICITY']
    icu_pat_admit = icu_pat_admit.apply(lambda x: x.mask(x.map(x.value_counts()) < 100, 'OTHER/UNKNOWN') if x.name in columns_to_mask else x)
    icu_pat_admit = icu_pat_admit.apply(lambda x: x.str.title() if x.name in columns_to_mask else x)

    # Compute pre-ICU length of stay in fractional days
    icu_pat_admit['PRE_ICU_LOS'] = (icu_pat_admit['INTIME'] - icu_pat_admit['ADMITTIME']) / np.timedelta64(1, 'D')
    icu_pat_admit.loc[icu_pat_admit['PRE_ICU_LOS']<0, 'PRE_ICU_LOS'] = 0

    # Clean up
    icu_pat_admit.drop(columns=['ADMITTIME'], inplace=True)

    print('-----------------------------------------')

    # Link services table
    # Keep first service only
    services.sort_values(by=['HADM_ID', 'TRANSFERTIME'], ascending=True, inplace=True)
    services = services.groupby(['HADM_ID']).nth(0).reset_index()

    # Check if first service is a surgery
    services['SURGERY'] = services['CURR_SERVICE'].str.contains('SURG') | (services['CURR_SERVICE'] == 'ORTHO')

    print('Link services table...')
    icu_pat_admit = pd.merge(icu_pat_admit, services, how='left', on=['SUBJECT_ID', 'HADM_ID'])

    # Get elective surgery admissions
    icu_pat_admit['ELECTIVE_SURGERY'] = ((icu_pat_admit['ADMISSION_TYPE'] == 'ELECTIVE') & icu_pat_admit['SURGERY']).astype(int)

    # Clean up
    icu_pat_admit.drop(columns=['TRANSFERTIME', 'CURR_SERVICE', 'ADMISSION_TYPE', 'SURGERY'], inplace=True)

    print('Save...')
    icu_pat_admit.sort_values(by='ICUSTAY_ID', ascending=True, inplace=True)
    icu_pat_admit.to_pickle(data_dir + 'icu_pat_admit.pkl')
    icu_pat_admit.to_csv(data_dir + 'icu_pat_admit.csv', index=False)

def process_diagnoses_procedures():
    print('Loading icu_pat...')
    icu_pat = pd.read_pickle(data_dir + 'icu_pat_admit.pkl')

    print('-----------------------------------------')
    print('Load admissions...')
    # Load admissions table
    # Table purpose: Define a patients hospital admission, HADM_ID.
    dtype = {'HADM_ID': 'int32',
             'ADMITTIME': 'str',
             'DISCHTIME': 'str'}
    parse_dates = ['ADMITTIME', 'DISCHTIME']
    admissions = pd.read_csv(csv_dir + 'ADMISSIONS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)

    print('-----------------------------------------')
    print('Load diagnoses and procedures...')
    dtype = {'SUBJECT_ID': 'int32',
             'HADM_ID': 'int32',
             'ICD9_CODE': 'str'}
    # Load diagnosis_icd table
    # Table purpose: Contains ICD diagnoses for patients, most notably ICD-9 diagnoses.
    diagnoses = pd.read_csv(csv_dir + 'DIAGNOSES_ICD.csv', usecols=dtype.keys(), dtype=dtype)
    diagnoses = diagnoses.dropna()
    # Load procedures_icd table
    # Table purpose: Contains ICD procedures for patients, most notably ICD-9 procedures.
    procedures = pd.read_csv(csv_dir + 'PROCEDURES_ICD.csv', usecols=dtype.keys(), dtype=dtype)
    procedures = procedures.dropna()

    # Merge diagnoses and procedures
    diagnoses['ICD9_CODE'] = 'DIAGN_' + diagnoses['ICD9_CODE'].str.lower().str.strip()
    procedures['ICD9_CODE'] = 'PROCE_' + procedures['ICD9_CODE'].str.lower().str.strip()
    diag_proc = pd.concat([diagnoses, procedures], ignore_index=True, sort=False)

    print('-----------------------------------------')

    # Link diagnoses/procedures and admissions tables
    print('Link diagnoses/procedures and admissions tables...')
    diag_proc = pd.merge(diag_proc, admissions, how='inner', on='HADM_ID').drop(columns=['HADM_ID'])

    # Link diagnoses/procedures and icu_pat tables
    print('Link diagnoses/procedures and icu_pat tables...')
    diag_proc = pd.merge(icu_pat[['SUBJECT_ID', 'ICUSTAY_ID', 'OUTTIME']], diag_proc, how='left', on=['SUBJECT_ID'])

    # Remove codes related to future admissions using time difference to ADMITTIME
    diag_proc['DAYS_TO_OUT'] = (diag_proc['OUTTIME']-diag_proc['ADMITTIME']) / np.timedelta64(1, 'D')
    diag_proc = diag_proc[(diag_proc['DAYS_TO_OUT'] >= 0) | diag_proc['DAYS_TO_OUT'].isna()]
    # Reset time value using time difference to DISCHTIME (0 if negative)
    diag_proc['DAYS_TO_OUT'] = (diag_proc['OUTTIME']-diag_proc['DISCHTIME']) / np.timedelta64(1, 'D')
    diag_proc.loc[diag_proc['DAYS_TO_OUT'] < 0, 'DAYS_TO_OUT'] = 0
    diag_proc = diag_proc.drop(columns=['SUBJECT_ID', 'OUTTIME', 'ADMITTIME', 'DISCHTIME'])
    # Lost some ICUSTAY_IDs with only negative DAYS_TO_OUT, merge back
    diag_proc = pd.merge(icu_pat[['ICUSTAY_ID']], diag_proc, how='left', on=['ICUSTAY_ID'])

    print('Drop duplicates...')
    diag_proc = diag_proc.drop_duplicates()

    print('Map rare codes to OTHER...')
    diag_proc = diag_proc.apply(lambda x: x.mask(x.map(x.value_counts()) < 100, 'other') if x.name in ['ICD9_CODE'] else x)

    print('-----------------------------------------')
    print('Save...')
    assert len(diag_proc['ICUSTAY_ID'].unique()) == 45298
    diag_proc.sort_values(by=['ICUSTAY_ID', 'DAYS_TO_OUT'], ascending=[True, True], inplace=True)
    diag_proc.to_pickle(data_dir + 'diag_proc.pkl')
    diag_proc.to_csv(data_dir + 'diag_proc.csv', index=False)

def process_charts():
    # Relevant ITEMIDs
    gcs_eye_opening          = [184, 220739, 226756, 227011]
    gcs_verbal_response      = [723, 223900, 226758, 227014]
    gcs_motor_response       = [454, 223901, 226757, 227012]
    gcs_total                = [198, 226755]
    diastolic_blood_pressure = [8364, 8368, 8440, 8441, 8502, 8503, 8506, 8555, 220051, 220180, 224643, 225310, 227242]
    systolic_blood_pressure  = [   6,   51,  442,  455, 3313, 3315, 3321, 6701, 220050, 220179, 224167, 225309, 227243]
    mean_blood_pressure      = [52, 443, 456, 2293, 2294, 2647, 3312, 3314, 3320, 6590, 6702, 6927, 7620, 220052, 220181, 225312]
    heart_rate               = [211, 220045, 227018]
    fraction_inspired_oxygen = [189, 190, 727, 1040, 1206, 1863, 2518, 2981, 3420, 3422, 7018, 7041, 7570, 223835, 226754, 227009, 227010]
    respiratory_rate         = [614, 615, 618, 619, 651, 653, 1884, 3603, 6749, 7884, 8113, 220210, 224422, 224688, 224689, 224690, 226774, 227050]
    body_temperature         = [676, 677, 678, 679, 3652, 3654, 6643, 223761, 223762, 226778, 227054]
    weight                   = [763, 3580, 3581, 3582, 3693, 224639, 226512, 226531]
    height                   = [1394, 226707, 226730]
    def inch_to_cm(value):
        return value*2.54

    def lb_to_kg(value):
        return value/2.205

    def oz_to_kg(value):
        return value/35.274

    def f_to_c(value):
        return (value-32)*5/9

    def frac_to_perc(value):
        return value*100

    # Relevant ITEMIDs
    body_temperature_F       = [678, 679, 3652, 3654, 6643, 223761, 226778, 227054]
    weight_lb                = [3581, 226531]
    weight_oz                = [3582]
    height_inch              = [1394, 226707]

    relevant_ids = (gcs_eye_opening + gcs_verbal_response + gcs_motor_response + gcs_total + mean_blood_pressure +
                    heart_rate + fraction_inspired_oxygen + respiratory_rate + body_temperature + weight + height)

    print('-----------------------------------------')
    print('Load item definitions')
    dtype = {'ITEMID': 'int32',
             'LABEL': 'str',
             'UNITNAME': 'str'}
    defs = pd.read_csv(csv_dir + 'D_ITEMS.csv', usecols=dtype.keys(), dtype=dtype)
    print('-----------------------------------------')

    print('Loading Chart Events')
    dtype = {'SUBJECT_ID': 'int32',
             'HADM_ID': 'int32',
             'ICUSTAY_ID': 'str',
             'ITEMID': 'int32',
             'CHARTTIME': 'str',
             'VALUENUM': 'float32'}
    parse_dates = ['CHARTTIME']
    # Load chartevents table
    # Table purpose: Contains all charted data for all patients.
    chunksize = 1000000
    i = 0
    # Not parsing dates
    for df in tqdm(pd.read_csv(csv_dir + 'CHARTEVENTS.csv', usecols=dtype.keys(), dtype=dtype, chunksize=chunksize)):
        df = df[df['ICUSTAY_ID'].notna() & df['VALUENUM'].notna() & (df['ITEMID'].isin(relevant_ids)) & (df['VALUENUM'] > 0)]
        # convert units
        df.loc[df['ITEMID'].isin(body_temperature_F), 'VALUENUM'] = f_to_c(df[df['ITEMID'].isin(body_temperature_F)].VALUENUM)
        df.loc[df['ITEMID'].isin(weight_lb), 'VALUENUM'] = lb_to_kg(df[df['ITEMID'].isin(weight_lb)].VALUENUM)
        df.loc[df['ITEMID'].isin(weight_oz), 'VALUENUM'] = oz_to_kg(df[df['ITEMID'].isin(weight_oz)].VALUENUM)
        df.loc[df['ITEMID'].isin(height_inch), 'VALUENUM'] = inch_to_cm(df[df['ITEMID'].isin(height_inch)].VALUENUM)
        df.loc[(df['ITEMID'].isin(fraction_inspired_oxygen)) & (df['VALUENUM']<=1), 'VALUENUM'] = frac_to_perc(df[(df['ITEMID'].isin(fraction_inspired_oxygen)) & (df['VALUENUM']<=1)].VALUENUM)
        # remove implausible measurements
        df = df[~(df['ITEMID'].isin(gcs_total) & (df.VALUENUM < 3))]
        df = df[~(df['ITEMID'].isin(diastolic_blood_pressure + systolic_blood_pressure + mean_blood_pressure) & (df.VALUENUM > 250))]
        df = df[~(df['ITEMID'].isin(heart_rate) & ((df.VALUENUM < 1) | (df.VALUENUM > 250)))]
        df = df[~(df['ITEMID'].isin(fraction_inspired_oxygen) & (df.VALUENUM > 100))]
        df = df[~(df['ITEMID'].isin(respiratory_rate) & ((df.VALUENUM < 1) | (df.VALUENUM > 100)))]
        df = df[~(df['ITEMID'].isin(body_temperature) & (df.VALUENUM > 50))]
        df = df[~(df['ITEMID'].isin(weight) & (df.VALUENUM > 700))]
        df = df[~(df['ITEMID'].isin(height) & (df.VALUENUM > 300))]
        df = df[df['VALUENUM'] > 0]
        # label
        df['CE_TYPE'] = ''
        df.loc[df['ITEMID'].isin(gcs_eye_opening), 'CE_TYPE'] = 'GCS_EYE_OPENING'
        df.loc[df['ITEMID'].isin(gcs_verbal_response), 'CE_TYPE'] = 'GCS_VERBAL_RESPONSE'
        df.loc[df['ITEMID'].isin(gcs_motor_response), 'CE_TYPE'] = 'GCS_MOTOR_RESPONSE'
        df.loc[df['ITEMID'].isin(gcs_total), 'CE_TYPE'] = 'GCS_TOTAL'
        df.loc[df['ITEMID'].isin(diastolic_blood_pressure), 'CE_TYPE'] = 'DIASTOLIC_BP'
        df.loc[df['ITEMID'].isin(systolic_blood_pressure), 'CE_TYPE'] = 'SYSTOLIC_BP'
        df.loc[df['ITEMID'].isin(mean_blood_pressure), 'CE_TYPE'] = 'MEAN_BP'
        df.loc[df['ITEMID'].isin(heart_rate), 'CE_TYPE'] = 'HEART_RATE'
        df.loc[df['ITEMID'].isin(fraction_inspired_oxygen), 'CE_TYPE'] = 'FRACTION_INSPIRED_OXYGEN'
        df.loc[df['ITEMID'].isin(respiratory_rate), 'CE_TYPE'] = 'RESPIRATORY_RATE'
        df.loc[df['ITEMID'].isin(body_temperature), 'CE_TYPE'] = 'BODY_TEMPERATURE'
        df.loc[df['ITEMID'].isin(weight), 'CE_TYPE'] = 'WEIGHT'
        df.loc[df['ITEMID'].isin(height), 'CE_TYPE'] = 'HEIGHT'
        df.drop(columns=['ITEMID'], inplace=True)

        # save
        if i == 0:
            df.to_csv(data_dir + 'chartevents_reduced.csv', index=False)
        else:
            df.to_csv(data_dir + 'chartevents_reduced.csv', mode='a', header=False, index=False)
        i += 1

def process_output():
    urine_output = [42810, 43171, 43173, 43175, 43348, 43355, 43365, 43372, 43373, 43374, 43379, 43380, 43431, 43462, 43522, 40405, 40428, 40534,
                    40288, 42042, 42068, 42111, 42119, 42209, 41857, 40715, 40056, 40061, 40085, 40094, 40096, 42001, 42676, 42556, 43093, 44325, 44706,
                    44506, 42859, 44237, 44313, 44752, 44824, 44837, 43576, 43589, 43633, 44911, 44925, 42362, 42463, 42507, 42510, 40055, 40057, 40065,
                    40069, 45804, 45841, 43811, 43812, 43856, 43897, 43931, 43966, 44080, 44103, 44132, 45304, 46177, 46532, 46578, 46658, 46748, 40651,
                    43053, 43057, 40473, 42130, 41922, 44253, 44278, 46180, 44684, 43333, 43347, 42592, 42666, 42765, 42892, 45927, 44834, 43638, 43654,
                    43519, 43537, 42366, 45991, 46727, 46804, 43987, 44051, 227489, 226566, 226627, 226631, 45415, 42111, 41510, 40055, 226559, 40428,
                    40580, 40612, 40094, 40848, 43685, 42362, 42463, 42510, 46748, 40972, 40973, 46456, 226561, 226567, 226632, 40096, 40651, 226557,
                    226558, 40715, 226563]
    # Relevant ITEMIDs
    print('-----------------------------------------')
    print('Load item definitions')
    dtype = {'ITEMID': 'int32',
             'LABEL': 'str',
             'UNITNAME': 'str',
             'LINKSTO': 'str'}
    defs = pd.read_csv(csv_dir + 'D_ITEMS.csv', usecols=dtype.keys(), dtype=dtype)
    print('URINE_OUTPUT')
    defs = defs[defs['ITEMID'].isin(urine_output)]
    defs['LABEL'] = defs['LABEL'].str.lower()
    # Remove measurements in /kg/hr
    defs = defs[~(defs['LABEL'].str.contains('hr') | defs['LABEL'].str.contains('kg')) | defs['LABEL'].str.contains('nephro')]
    print(defs['LABEL'])
    urine_output = defs['ITEMID'].tolist()
    print('-----------------------------------------')

    print('Loading Output Events')
    dtype = {'ICUSTAY_ID': 'str',
             'ITEMID': 'int32',
             'CHARTTIME': 'str',
             'VALUE': 'float32'}
    parse_dates = ['CHARTTIME']

    # Load outputevents table
    # Table purpose: Output data for patients.
    df = pd.read_csv(csv_dir + 'OUTPUTEVENTS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
    df = df.rename(columns={'VALUE': 'VALUENUM'})
    df = df[df['ICUSTAY_ID'].notna() & df['VALUENUM'].notna() & (df['ITEMID'].isin(urine_output)) & (df['VALUENUM'] > 0)]
    df['ICUSTAY_ID'] = df['ICUSTAY_ID'].astype('int32')

    # remove implausible measurements
    df = df[~(df.VALUENUM > 10000)]

    # sum all outputs in one day
    df.drop(columns=['ITEMID'], inplace=True)
    df['CHARTTIME'] = df['CHARTTIME'].dt.date
    df = df.groupby(['ICUSTAY_ID', 'CHARTTIME']).sum()
    df['CE_TYPE'] = 'URINE_OUTPUT'
    df = df[~(df.VALUENUM > 10000)]

    print('Remove admission and discharge days (since data on urine output is incomplete)')
    # Load icustays table
    # Table purpose: Defines each ICUSTAY_ID in the database, i.e. defines a single ICU stay
    print('Load ICU stays...')
    dtype = {'ICUSTAY_ID': 'int32',
             'INTIME': 'str',
             'OUTTIME': 'str'}
    parse_dates = ['INTIME', 'OUTTIME']
    icustays = pd.read_csv(csv_dir + 'ICUSTAYS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
    icustays['INTIME'] = icustays['INTIME'].dt.date
    icustays['OUTTIME'] = icustays['OUTTIME'].dt.date

    # Merge
    tmp = icustays[['ICUSTAY_ID', 'INTIME']].drop_duplicates()
    tmp = tmp.rename(columns={'INTIME': 'CHARTTIME'})
    tmp['ID_IN'] = 1
    df = pd.merge(df, tmp, how='left', on=['ICUSTAY_ID', 'CHARTTIME'])
    tmp = icustays[['ICUSTAY_ID', 'OUTTIME']].drop_duplicates()
    tmp = tmp.rename(columns={'OUTTIME': 'CHARTTIME'})
    tmp['ID_OUT'] = 1
    df = pd.merge(df, tmp, how='left', on=['ICUSTAY_ID', 'CHARTTIME'])

    # Remove admission and discharge days
    df = df[df['ID_IN'].isnull() & df['ID_OUT'].isnull()]
    df.drop(columns=['ID_IN', 'ID_OUT'], inplace=True)

    # Add SUBJECT_ID and HADM_ID
    icustays.drop(columns=['INTIME', 'OUTTIME'], inplace=True)
    df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME']) + pd.DateOffset(hours=12)

    # Save
    df.to_pickle(data_dir + 'outputevents_reduced.pkl')

def merge_charts_output():
    print('Loading chart events...')
    dtype = {'SUBJECT_ID': 'int32',
             'ICUSTAY_ID': 'int32',
             'CE_TYPE': 'str',
             'CHARTTIME': 'str',
             'VALUENUM': 'float32'}
    parse_dates = ['CHARTTIME']
    charts = pd.read_csv(data_dir + 'chartevents_reduced.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)

    print('-----------------------------------------')

    print('Compute BMI and GCS total...')
    charts.sort_values(by=['SUBJECT_ID', 'ICUSTAY_ID', 'CHARTTIME'], ascending=[True, True, False], inplace=True)

    # Compute BMI
    rows_bmi = (charts['CE_TYPE']=='WEIGHT') | (charts['CE_TYPE']=='HEIGHT')
    charts_bmi = charts[rows_bmi]
    charts_bmi = charts_bmi.pivot_table(index=['SUBJECT_ID', 'ICUSTAY_ID', 'CHARTTIME'], columns='CE_TYPE', values='VALUENUM')
    charts_bmi = charts_bmi.rename_axis(None, axis=1).reset_index()
    charts_bmi['HEIGHT'] = charts_bmi.groupby('SUBJECT_ID')['HEIGHT'].ffill()
    charts_bmi['HEIGHT'] = charts_bmi.groupby('SUBJECT_ID')['HEIGHT'].bfill()
    charts_bmi =  charts_bmi[~pd.isnull(charts_bmi).any(axis=1)]
    charts_bmi['VALUENUM'] = charts_bmi['WEIGHT']/charts_bmi['HEIGHT']/charts_bmi['HEIGHT']*10000
    charts_bmi['CE_TYPE'] = 'BMI'
    charts_bmi.drop(columns=['HEIGHT', 'WEIGHT'], inplace=True)

    # Compute GCS total if not available
    rows_gcs = (charts['CE_TYPE']=='GCS_EYE_OPENING') | (charts['CE_TYPE']=='GCS_VERBAL_RESPONSE') | (charts['CE_TYPE']=='GCS_MOTOR_RESPONSE') | (charts['CE_TYPE']=='GCS_TOTAL')
    charts_gcs = charts[rows_gcs]
    charts_gcs = charts_gcs.pivot_table(index=['SUBJECT_ID', 'ICUSTAY_ID', 'CHARTTIME'], columns='CE_TYPE', values='VALUENUM')
    charts_gcs = charts_gcs.rename_axis(None, axis=1).reset_index()
    null_gcs_total = charts_gcs['GCS_TOTAL'].isnull()
    charts_gcs.loc[null_gcs_total, 'GCS_TOTAL'] = charts_gcs[null_gcs_total].GCS_EYE_OPENING + charts_gcs[null_gcs_total].GCS_VERBAL_RESPONSE + charts_gcs[null_gcs_total].GCS_MOTOR_RESPONSE
    charts_gcs =  charts_gcs[~charts_gcs['GCS_TOTAL'].isnull()]
    charts_gcs = charts_gcs.rename(columns={'GCS_TOTAL': 'VALUENUM'})
    charts_gcs['CE_TYPE'] = 'GCS_TOTAL'
    charts_gcs.drop(columns=['GCS_EYE_OPENING', 'GCS_VERBAL_RESPONSE', 'GCS_MOTOR_RESPONSE'], inplace=True)

    # Merge back with rest of the table
    rows_others = ~rows_bmi & ~rows_gcs
    charts = pd.concat([charts_bmi, charts_gcs, charts[rows_others]], ignore_index=True, sort=False)
    charts.drop(columns=['SUBJECT_ID'], inplace=True)
    charts.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False], inplace=True)

    print('-----------------------------------------')

    # Load (reduced) outputevents table
    print('Loading output events...')
    outputs = pd.read_pickle(data_dir + 'outputevents_reduced.pkl')
    df = pd.concat([charts, outputs], ignore_index=True, sort=False)
    df.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False], inplace=True)

    print('-----------------------------------------')

    print('Create categorical variable...')
    # Bin according to OASIS severity score
    heart_rate_bins               = np.array([-1, 32.99, 88.5, 106.5, 125.5, np.Inf])
    respiratory_rate_bins         = np.array([-1, 5.99, 12.5, 22.5, 30.5, 44.5, np.Inf])
    body_temperature_bins         = np.array([-1, 33.21, 35.93, 36.39, 36.88, 39.88, np.Inf])
    mean_bp_bins                  = np.array([-1, 20.64, 50.99, 61.32, 143.44, np.Inf])
    fraction_inspired_oxygen_bins = np.array([-1, np.Inf])
    gcs_total_bins                = np.array([-1, 7, 13, 14, 15])
    bmi_bins                      = np.array([-1, 15, 16, 18.5, 25, 30, 35, 40, 45, 50, 60, np.Inf])
    urine_output_bins             = np.array([-1, 670.99, 1426.99, 2543.99, 6896, np.Inf])
    #bins = [heart_rate_bins, respiratory_rate_bins, body_temperature_bins, mean_bp_bins, fraction_inspired_oxygen_bins, gcs_total_bins, bmi_bins, urine_output_bins]
    bins = [heart_rate_bins, respiratory_rate_bins, body_temperature_bins, mean_bp_bins, fraction_inspired_oxygen_bins, gcs_total_bins, urine_output_bins]
    # Labels
    heart_rate_labels               = ['CHART_HR_m1', 'CHART_HR_n', 'CHART_HR_p1', 'CHART_HR_p2', 'CHART_HR_p3']
    respiratory_rate_labels         = ['CHART_RR_m2', 'CHART_RR_m1', 'CHART_RR_n', 'CHART_RR_p1', 'CHART_RR_p2', 'CHART_RR_p3']
    body_temperature_labels         = ['CHART_BT_m3', 'CHART_BT_m2', 'CHART_BT_m1', 'CHART_BT_n', 'CHART_BT_p1', 'CHART_BT_p2']
    mean_bp_labels                  = ['CHART_BP_m3', 'CHART_BP_m2', 'CHART_BP_m1', 'CHART_BP_n', 'CHART_BP_p1']
    fraction_inspired_oxygen_labels = ['CHART_VENT']
    gcs_total_labels                = ['CHART_GC_m3', 'CHART_GC_m2', 'CHART_GC_m1', 'CHART_GC_n']
    bmi_labels                      = ['CHART_BM_m3', 'CHART_BM_m2', 'CHART_BM_m1', 'CHART_BM_n', 'CHART_BM_p1', 'CHART_BM_p2', 'CHART_BM_p3', 'CHART_BM_p4', 'CHART_BM_p5', 'CHART_BM_p6', 'CHART_BM_p7']
    urine_output_labels             = ['CHART_UO_m3', 'CHART_UO_m2', 'CHART_UO_m1', 'CHART_UO_n', 'CHART_UO_p1']
    #labels = [heart_rate_labels, respiratory_rate_labels, body_temperature_labels, mean_bp_labels, fraction_inspired_oxygen_labels, gcs_total_labels, bmi_labels, urine_output_labels]
    labels = [heart_rate_labels, respiratory_rate_labels, body_temperature_labels, mean_bp_labels, fraction_inspired_oxygen_labels, gcs_total_labels, urine_output_labels]
    # Chart event types
    #ce_types = ['HEART_RATE', 'RESPIRATORY_RATE', 'BODY_TEMPERATURE', 'MEAN_BP', 'FRACTION_INSPIRED_OXYGEN', 'GCS_TOTAL', 'BMI', 'URINE_OUTPUT']
    ce_types = ['HEART_RATE', 'RESPIRATORY_RATE', 'BODY_TEMPERATURE', 'MEAN_BP', 'FRACTION_INSPIRED_OXYGEN', 'GCS_TOTAL', 'URINE_OUTPUT']

    df_list = []
    df_list_last_only = [] # for logistic regression
    for type, label, bin in zip(ce_types, labels, bins):
        # get chart events of a specific type
        tmp = df[df['CE_TYPE'] == type]
        # bin them and sort
        tmp['VALUECAT'] = pd.cut(tmp['VALUENUM'], bins=bin, labels=label)
        tmp.drop(columns=['CE_TYPE', 'VALUENUM'], inplace=True)
        tmp.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False], inplace=True)
        # remove consecutive duplicates
        tmp = tmp[(tmp[['ICUSTAY_ID', 'VALUECAT']] != tmp[['ICUSTAY_ID', 'VALUECAT']].shift()).any(axis=1)]
        df_list.append(tmp)
        # for logistic regression, keep only the last measurement
        tmp = tmp.drop_duplicates(subset='ICUSTAY_ID')
        df_list_last_only.append(tmp)

    df = pd.concat(df_list, ignore_index=True, sort=False)
    df.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False], inplace=True)

    # drop duplicates to keep size manageable
    df = df.drop_duplicates()

    print('-----------------------------------------')

    print('Save...')
    df.to_pickle(data_dir + 'charts_outputs_reduced.pkl')
    df.to_csv(data_dir + 'charts_outputs_reduced.csv', index=False)

    print('-----------------------------------------')

    print('Save data for logistic regression...')

    # for logistic regression
    df_last_only = pd.concat(df_list_last_only, ignore_index=True, sort=False)
    df_last_only.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False], inplace=True)
    df_last_only.to_pickle(data_dir + 'charts_outputs_last_only.pkl')
    df_last_only.to_csv(data_dir + 'charts_outputs_last_only.csv', index=False)

def process_charts_prescriptions():
    # Load icu_pat table
    print('Loading icu_pat...')
    icu_pat = pd.read_pickle(data_dir + 'icu_pat_admit.pkl')

    print('-----------------------------------------')
    print('Load charts and outputs...')
    charts_outputs = pd.read_pickle(data_dir + 'charts_outputs_reduced.pkl')

    print('-----------------------------------------')
    print('Load prescriptions...')
    dtype = {'ICUSTAY_ID': 'str',
             'DRUG': 'str',
             'STARTDATE': 'str'}
    parse_dates = ['STARTDATE']
    # Load prescriptions table
    # Table purpose: Contains medication related order entries, i.e. prescriptions
    prescriptions = pd.read_csv(csv_dir + 'PRESCRIPTIONS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
    prescriptions = prescriptions.dropna()
    prescriptions['ICUSTAY_ID'] = prescriptions['ICUSTAY_ID'].astype('int32')
    prescriptions['DRUG'] = 'PRESC_' + prescriptions['DRUG'].str.lower().replace('\s+', '', regex=True)
    prescriptions = prescriptions.rename(columns={'DRUG': 'VALUECAT', 'STARTDATE': 'CHARTTIME'})
    df = pd.concat([charts_outputs, prescriptions], ignore_index=True, sort=False)

    print('-----------------------------------------')

    # Link charts/outputs and icu_pat tables
    print('Link charts/outputs and icu_pat tables...')
    df = pd.merge(icu_pat[['ICUSTAY_ID', 'OUTTIME']], df, how='left', on=['ICUSTAY_ID'])

    # Reset time value using time difference to DISCHTIME (0 if negative)
    df['HOURS_TO_OUT'] = (df['OUTTIME']-df['CHARTTIME']) / np.timedelta64(1, 'h')
    df.loc[df['HOURS_TO_OUT'] < 0, 'HOURS_TO_OUT'] = 0
    df = df.drop(columns=['OUTTIME', 'CHARTTIME'])

    print('Drop duplicates...')
    df = df.drop_duplicates()

    print('Map rare codes to OTHER...')
    df = df.apply(lambda x: x.mask(x.map(x.value_counts()) < 100, 'other') if x.name in ['VALUECAT'] else x)

    print('-----------------------------------------')
    print('Save...')
    assert len(df['ICUSTAY_ID'].unique()) == 45298
    df.sort_values(by=['ICUSTAY_ID', 'HOURS_TO_OUT'], ascending=[True, True], inplace=True)
    df.to_pickle(data_dir + 'charts_prescriptions.pkl')
    df.to_csv(data_dir + 'charts_prescriptions.csv', index=False)

def get_arrays(df, icu_pat, code_column, time_column, quantile=1):
    df['COUNT'] = df.groupby(['ICUSTAY_ID']).cumcount()
    df = df[df['COUNT'] < df.groupby(['ICUSTAY_ID']).size().quantile(q=quantile)]
    max_count_df = df['COUNT'].max()+1
    print('max_count {}'.format(max_count_df))
    multiindex_df = pd.MultiIndex.from_product([icu_pat['ICUSTAY_ID'], range(max_count_df)], names = ['ICUSTAY_ID', 'COUNT'])
    df = df.set_index(['ICUSTAY_ID', 'COUNT'])

    print('Reindex df...')
    df = df.reindex(multiindex_df).fillna(0)
    print('done')
    df_times = df[time_column].values.reshape((len(icu_pat['ICUSTAY_ID']), max_count_df))
    df[code_column] = df[code_column].astype('category')
    dict_df = dict(enumerate(df[code_column].cat.categories))
    df[code_column] = df[code_column].cat.codes
    df = df[code_column].values.reshape((len(icu_pat['ICUSTAY_ID']), max_count_df))

    return df, df_times, dict_df

def create_arrays():
    # Load icu_pat table
    print('Loading icu_pat...')
    icu_pat = pd.read_pickle(data_dir + 'icu_pat_admit.pkl')

    print('Loading diagnoses/procedures...')
    dp = pd.read_pickle(data_dir + 'diag_proc.pkl')

    print('Loading charts/prescriptions...')
    cp = pd.read_pickle(data_dir + 'charts_prescriptions.pkl')

    print('-----------------------------------------')

    # static variables
    print('Create static array...')
    icu_pat = pd.get_dummies(icu_pat, columns = ['ADMISSION_LOCATION', 'INSURANCE', 'MARITAL_STATUS', 'ETHNICITY'])
    icu_pat.drop(columns=['ADMISSION_LOCATION_Emergency Room Admit', 'INSURANCE_Medicare', 'MARITAL_STATUS_Married/Life Partner', 'ETHNICITY_White'], inplace=True) # drop reference columns
    static_columns = icu_pat.columns.str.contains('AGE|GENDER_M|LOS|NUM_RECENT_ADMISSIONS|ADMISSION_LOCATION|INSURANCE|MARITAL_STATUS|ETHNICITY|PRE_ICU_LOS|ELECTIVE_SURGERY')
    static = icu_pat.loc[:, static_columns].values
    static_vars = icu_pat.loc[:, static_columns].columns.values.tolist()

    # classification label
    print('Create label array...')
    label = icu_pat.loc[:, 'POSITIVE'].values

    # diagnoses/procedures and charts/prescriptions
    print('Create diagnoses/procedures and charts/prescriptions array...')
    dp, dp_times, dict_dp = get_arrays(dp, icu_pat, 'ICD9_CODE', 'DAYS_TO_OUT', 1)
    cp, cp_times, dict_cp = get_arrays(cp, icu_pat, 'VALUECAT', 'HOURS_TO_OUT', 0.95)

    # Normalize times
    dp_times = dp_times/dp_times.max()
    cp_times = cp_times/cp_times.max()

    print('-----------------------------------------')

    print('Split data into train/validate/test...')
    # Split patients to avoid data leaks
    patients = icu_pat['SUBJECT_ID'].drop_duplicates()
    train, validate, test = np.split(patients.sample(frac=1, random_state=123), [int(.9*len(patients)), int(.9*len(patients))])
    train_ids = icu_pat['SUBJECT_ID'].isin(train).values
    validate_ids = icu_pat['SUBJECT_ID'].isin(validate).values
    test_ids = icu_pat['SUBJECT_ID'].isin(test).values

    print('Get patients corresponding to test ids')
    test_ids_patients = icu_pat['SUBJECT_ID'].iloc[test_ids].reset_index(drop=True)

    print('-----------------------------------------')

    print('Save...')
    np.savez(data_dir + 'data_arrays.npz', static=static, static_vars=static_vars, label=label,
             dp=dp, cp=cp, dp_times=dp_times, cp_times=cp_times, dict_dp=dict_dp, dict_cp=dict_cp,
             train_ids=train_ids, validate_ids=validate_ids, test_ids=test_ids)
    np.savez(data_dir + 'data_dictionaries.npz', dict_dp=dict_dp, dict_cp=dict_cp)
    test_ids_patients.to_pickle(data_dir + 'test_ids_patients.pkl')

if __name__ == '__main__':
    process_admission()
    process_diagnoses_procedures()
    process_charts()
    process_output()
    merge_charts_output()
    process_charts_prescriptions()
    create_arrays()