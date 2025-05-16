def R134a_diag():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np


    filepath= 'Refrigerants/'+'R134a'+'.csv'

    R134a_saturated_input_data=pd.read_csv(filepath, encoding='cp1252')
    R134a_saturated_input_data['DRUCK']=R134a_saturated_input_data['DRUCK']/100000
    R134a_saturated_input_data['Enthalpy_Quality0.1']=R134a_saturated_input_data['ENTHA0']+0.1*(R134a_saturated_input_data['ENTHA1']-R134a_saturated_input_data['ENTHA0'])
    R134a_saturated_input_data['Enthalpy_Quality0.2']=R134a_saturated_input_data['ENTHA0']+0.2*(R134a_saturated_input_data['ENTHA1']-R134a_saturated_input_data['ENTHA0'])
    R134a_saturated_input_data['Enthalpy_Quality0.3']=R134a_saturated_input_data['ENTHA0']+0.3*(R134a_saturated_input_data['ENTHA1']-R134a_saturated_input_data['ENTHA0'])
    R134a_saturated_input_data['Enthalpy_Quality0.4']=R134a_saturated_input_data['ENTHA0']+0.4*(R134a_saturated_input_data['ENTHA1']-R134a_saturated_input_data['ENTHA0'])
    R134a_saturated_input_data['Enthalpy_Quality0.5']=R134a_saturated_input_data['ENTHA0']+0.5*(R134a_saturated_input_data['ENTHA1']-R134a_saturated_input_data['ENTHA0'])
    R134a_saturated_input_data['Enthalpy_Quality0.6']=R134a_saturated_input_data['ENTHA0']+0.6*(R134a_saturated_input_data['ENTHA1']-R134a_saturated_input_data['ENTHA0'])
    R134a_saturated_input_data['Enthalpy_Quality0.7']=R134a_saturated_input_data['ENTHA0']+0.7*(R134a_saturated_input_data['ENTHA1']-R134a_saturated_input_data['ENTHA0'])
    R134a_saturated_input_data['Enthalpy_Quality0.8']=R134a_saturated_input_data['ENTHA0']+0.8*(R134a_saturated_input_data['ENTHA1']-R134a_saturated_input_data['ENTHA0'])
    R134a_saturated_input_data['Enthalpy_Quality0.9']=R134a_saturated_input_data['ENTHA0']+0.9*(R134a_saturated_input_data['ENTHA1']-R134a_saturated_input_data['ENTHA0'])

    max_enthal_index = R134a_saturated_input_data['Enthalpy_Quality0.1'].idxmax()
    R134a_saturated_input_data_1_1 = R134a_saturated_input_data.loc[:max_enthal_index]
    R134a_saturated_input_data_1_2 = R134a_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R134a_saturated_input_data['Enthalpy_Quality0.2'].idxmax()
    R134a_saturated_input_data_2_1 = R134a_saturated_input_data.loc[:max_enthal_index]
    R134a_saturated_input_data_2_2 = R134a_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R134a_saturated_input_data['Enthalpy_Quality0.3'].idxmax()
    R134a_saturated_input_data_3_1 = R134a_saturated_input_data.loc[:max_enthal_index]
    R134a_saturated_input_data_3_2 = R134a_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R134a_saturated_input_data['Enthalpy_Quality0.4'].idxmax()
    R134a_saturated_input_data_4_1 = R134a_saturated_input_data.loc[:max_enthal_index]
    R134a_saturated_input_data_4_2 = R134a_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R134a_saturated_input_data['Enthalpy_Quality0.5'].idxmax()
    R134a_saturated_input_data_5_1 = R134a_saturated_input_data.loc[:max_enthal_index]
    R134a_saturated_input_data_5_2 = R134a_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R134a_saturated_input_data['Enthalpy_Quality0.6'].idxmax()
    R134a_saturated_input_data_6_1 = R134a_saturated_input_data.loc[:max_enthal_index]
    R134a_saturated_input_data_6_2 = R134a_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R134a_saturated_input_data['Enthalpy_Quality0.7'].idxmax()
    R134a_saturated_input_data_7_1 = R134a_saturated_input_data.loc[:max_enthal_index]
    R134a_saturated_input_data_7_2 = R134a_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R134a_saturated_input_data['Enthalpy_Quality0.8'].idxmax()
    R134a_saturated_input_data_8_1 = R134a_saturated_input_data.loc[:max_enthal_index]
    R134a_saturated_input_data_8_2 = R134a_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R134a_saturated_input_data['Enthalpy_Quality0.9'].idxmax()
    R134a_saturated_input_data_9_1 = R134a_saturated_input_data.loc[:max_enthal_index]
    R134a_saturated_input_data_9_2 = R134a_saturated_input_data.loc[max_enthal_index:]


    filepath='Refrigerants/'+'R134a'+'_isotherms_saturation.csv'
    R134a_isotherms=pd.read_csv(filepath, encoding='cp1252')
    R134a_isotherms['Pressure (Pa)']=R134a_isotherms['Pressure (Pa)']/100000

    R134a_isotherm_minus40=R134a_isotherms[R134a_isotherms['Temperature']==-40]
    R134a_isotherm_minus20=R134a_isotherms[R134a_isotherms['Temperature']==-20]
    R134a_isotherm_0=R134a_isotherms[R134a_isotherms['Temperature']==0]
    R134a_isotherm_20=R134a_isotherms[R134a_isotherms['Temperature']==20]
    R134a_isotherm_40=R134a_isotherms[R134a_isotherms['Temperature']==40]
    R134a_isotherm_60=R134a_isotherms[R134a_isotherms['Temperature']==60]
    R134a_isotherm_80=R134a_isotherms[R134a_isotherms['Temperature']==80]
    R134a_isotherm_100=R134a_isotherms[R134a_isotherms['Temperature']==100]
    R134a_isotherm_120=R134a_isotherms[R134a_isotherms['Temperature']==120]
    R134a_isotherm_140=R134a_isotherms[R134a_isotherms['Temperature']==140]
    R134a_isotherm_160=R134a_isotherms[R134a_isotherms['Temperature']==160]


    series_1 = pd.Series(R134a_saturated_input_data['ENTHA0'])
    series_2 = pd.Series(R134a_saturated_input_data['ENTHA1'])
    series_2=series_2[::-1]

    Enthal = pd.concat([series_1, series_2], ignore_index=True)

    series_a = pd.Series(R134a_saturated_input_data['DRUCK'])
    series_b = pd.Series(R134a_saturated_input_data['DRUCK'])
    series_b=series_b[::-1]
    DRUCK = pd.concat([series_a, series_b], ignore_index=True)
    concatenated_df = pd.concat([Enthal, DRUCK], axis=1)
    concatenated_df.rename(columns={0:'Enthalpy','DRUCK': 'Pressure'}, inplace=True)


    max_enthal_index = concatenated_df['Enthalpy'].idxmax()
    df1 = concatenated_df.loc[:max_enthal_index]
    df2 = concatenated_df.loc[max_enthal_index:]


    min_enthal_index = R134a_isotherm_minus40['Enthalpy (kJ/kg)'].idxmin()
    df_isotherm_minus40_1 = R134a_isotherm_minus40.loc[:min_enthal_index]
    df_isotherm_minus40_2 = R134a_isotherm_minus40.loc[min_enthal_index:]
    min_enthal_index = R134a_isotherm_minus20['Enthalpy (kJ/kg)'].idxmin()
    df_isotherm_minus20_1 = R134a_isotherm_minus20.loc[:min_enthal_index]
    df_isotherm_minus20_2 = R134a_isotherm_minus20.loc[min_enthal_index:]
    min_enthal_index = R134a_isotherm_0['Enthalpy (kJ/kg)'].idxmin()
    df_isotherm_0_1 = R134a_isotherm_0.loc[:min_enthal_index]
    df_isotherm_0_2 = R134a_isotherm_0.loc[min_enthal_index:]
    min_enthal_index = R134a_isotherm_20['Enthalpy (kJ/kg)'].idxmin()
    df_isotherm_20_1 = R134a_isotherm_20.loc[:min_enthal_index]
    df_isotherm_20_2 = R134a_isotherm_20.loc[min_enthal_index:]
    min_enthal_index = R134a_isotherm_40['Enthalpy (kJ/kg)'].idxmin()
    df_isotherm_40_1 = R134a_isotherm_40.loc[:min_enthal_index]
    df_isotherm_40_2 = R134a_isotherm_40.loc[min_enthal_index:]
    min_enthal_index = R134a_isotherm_60['Enthalpy (kJ/kg)'].idxmin()
    df_isotherm_60_1 = R134a_isotherm_60.loc[:min_enthal_index]
    df_isotherm_60_2 = R134a_isotherm_60.loc[min_enthal_index:]
    min_enthal_index = R134a_isotherm_80['Enthalpy (kJ/kg)'].idxmin()
    df_isotherm_80_1 = R134a_isotherm_80.loc[:min_enthal_index]
    df_isotherm_80_2 = R134a_isotherm_80.loc[min_enthal_index:]
    min_enthal_index = R134a_isotherm_100['Enthalpy (kJ/kg)'].idxmin()
    df_isotherm_100_1 = R134a_isotherm_100.loc[:min_enthal_index]
    df_isotherm_100_2 = R134a_isotherm_100.loc[min_enthal_index:]
    min_enthal_index = R134a_isotherm_120['Enthalpy (kJ/kg)'].idxmin()
    df_isotherm_120_1 = R134a_isotherm_120.loc[:min_enthal_index]
    df_isotherm_120_2 = R134a_isotherm_120.loc[min_enthal_index:]
    min_enthal_index = R134a_isotherm_140['Enthalpy (kJ/kg)'].idxmin()
    df_isotherm_140_1 = R134a_isotherm_140.loc[:min_enthal_index]
    df_isotherm_140_2 = R134a_isotherm_140.loc[min_enthal_index:]
    min_enthal_index = R134a_isotherm_160['Enthalpy (kJ/kg)'].idxmin()
    df_isotherm_160_1 = R134a_isotherm_160.loc[:min_enthal_index]
    df_isotherm_160_2 = R134a_isotherm_160.loc[min_enthal_index:]


    fig,ax = plt.subplots(figsize=(10.4,8.1))

    # Initial 4 coordinates (x and y)
    x_coords = np.array([0, 0, 0, 0])
    y_coords = np.array([0, 0, 0, 0])

    # Plot the initial 4 points
    points = ax.scatter(x_coords, y_coords, color='black',s=20)

    # Line connecting the 4 points in sequence (initially)
    line, = ax.plot(x_coords, y_coords, color='blue', linewidth=1)



    sns.lineplot(data=df_isotherm_minus40_1 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_minus40_2 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_minus20_1 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_minus20_2 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_0_1 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_0_2 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_20_1 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_20_2 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_40_1 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_40_2 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_60_1 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_60_2 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_80_1 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_80_2 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_100_1 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_100_2 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_120_1 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_120_2 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_140_1 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_140_2 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_160_1 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_160_2 , x='Enthalpy (kJ/kg)', y='Pressure (Pa)',linewidth=0.2,color='red')



    sns.lineplot(data=R134a_saturated_input_data_1_1, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_2_1, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_3_1, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_4_1, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_5_1, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_6_1, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_7_1, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_8_1, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_9_1, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_1_2, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_2_2, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_3_2, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_4_2, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_5_2, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_6_2, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_7_2, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_8_2, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R134a_saturated_input_data_9_2, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')

    sns.lineplot(data=df1, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')
    sns.lineplot(data=df2, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')

    # Set plot title and axis labels
    ax.set_title('R134a', fontsize=10)
    ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=10)
    ax.set_ylabel('Pressure [barA]', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(100, 550)  # Set x-axis limits
    ax.set_ylim(0.9, 200)  # Set y-axis limits
    ax.tick_params(axis='both', labelsize=10)


    plt.show(block=False)
    return(points,line,fig,ax)

def R1234yf_diag():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    import time


    filepath= 'Refrigerants/' + "R1234yf.csv"

    R1234yf_saturated_input_data=pd.read_csv(filepath, encoding='cp1252')
    R1234yf_saturated_input_data['DRUCK']=R1234yf_saturated_input_data['DRUCK']/100000
    R1234yf_saturated_input_data['Enthalpy_Quality0.1']=R1234yf_saturated_input_data['ENTHA0']+0.1*(R1234yf_saturated_input_data['ENTHA1']-R1234yf_saturated_input_data['ENTHA0'])
    R1234yf_saturated_input_data['Enthalpy_Quality0.2']=R1234yf_saturated_input_data['ENTHA0']+0.2*(R1234yf_saturated_input_data['ENTHA1']-R1234yf_saturated_input_data['ENTHA0'])
    R1234yf_saturated_input_data['Enthalpy_Quality0.3']=R1234yf_saturated_input_data['ENTHA0']+0.3*(R1234yf_saturated_input_data['ENTHA1']-R1234yf_saturated_input_data['ENTHA0'])
    R1234yf_saturated_input_data['Enthalpy_Quality0.4']=R1234yf_saturated_input_data['ENTHA0']+0.4*(R1234yf_saturated_input_data['ENTHA1']-R1234yf_saturated_input_data['ENTHA0'])
    R1234yf_saturated_input_data['Enthalpy_Quality0.5']=R1234yf_saturated_input_data['ENTHA0']+0.5*(R1234yf_saturated_input_data['ENTHA1']-R1234yf_saturated_input_data['ENTHA0'])
    R1234yf_saturated_input_data['Enthalpy_Quality0.6']=R1234yf_saturated_input_data['ENTHA0']+0.6*(R1234yf_saturated_input_data['ENTHA1']-R1234yf_saturated_input_data['ENTHA0'])
    R1234yf_saturated_input_data['Enthalpy_Quality0.7']=R1234yf_saturated_input_data['ENTHA0']+0.7*(R1234yf_saturated_input_data['ENTHA1']-R1234yf_saturated_input_data['ENTHA0'])
    R1234yf_saturated_input_data['Enthalpy_Quality0.8']=R1234yf_saturated_input_data['ENTHA0']+0.8*(R1234yf_saturated_input_data['ENTHA1']-R1234yf_saturated_input_data['ENTHA0'])
    R1234yf_saturated_input_data['Enthalpy_Quality0.9']=R1234yf_saturated_input_data['ENTHA0']+0.9*(R1234yf_saturated_input_data['ENTHA1']-R1234yf_saturated_input_data['ENTHA0'])

    max_enthal_index = R1234yf_saturated_input_data['Enthalpy_Quality0.1'].idxmax()
    R1234yf_saturated_input_data_1_1 = R1234yf_saturated_input_data.loc[:max_enthal_index]
    R1234yf_saturated_input_data_1_2 = R1234yf_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1234yf_saturated_input_data['Enthalpy_Quality0.2'].idxmax()
    R1234yf_saturated_input_data_2_1 = R1234yf_saturated_input_data.loc[:max_enthal_index]
    R1234yf_saturated_input_data_2_2 = R1234yf_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1234yf_saturated_input_data['Enthalpy_Quality0.3'].idxmax()
    R1234yf_saturated_input_data_3_1 = R1234yf_saturated_input_data.loc[:max_enthal_index]
    R1234yf_saturated_input_data_3_2 = R1234yf_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1234yf_saturated_input_data['Enthalpy_Quality0.4'].idxmax()
    R1234yf_saturated_input_data_4_1 = R1234yf_saturated_input_data.loc[:max_enthal_index]
    R1234yf_saturated_input_data_4_2 = R1234yf_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1234yf_saturated_input_data['Enthalpy_Quality0.5'].idxmax()
    R1234yf_saturated_input_data_5_1 = R1234yf_saturated_input_data.loc[:max_enthal_index]
    R1234yf_saturated_input_data_5_2 = R1234yf_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1234yf_saturated_input_data['Enthalpy_Quality0.6'].idxmax()
    R1234yf_saturated_input_data_6_1 = R1234yf_saturated_input_data.loc[:max_enthal_index]
    R1234yf_saturated_input_data_6_2 = R1234yf_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1234yf_saturated_input_data['Enthalpy_Quality0.7'].idxmax()
    R1234yf_saturated_input_data_7_1 = R1234yf_saturated_input_data.loc[:max_enthal_index]
    R1234yf_saturated_input_data_7_2 = R1234yf_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1234yf_saturated_input_data['Enthalpy_Quality0.8'].idxmax()
    R1234yf_saturated_input_data_8_1 = R1234yf_saturated_input_data.loc[:max_enthal_index]
    R1234yf_saturated_input_data_8_2 = R1234yf_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1234yf_saturated_input_data['Enthalpy_Quality0.9'].idxmax()
    R1234yf_saturated_input_data_9_1 = R1234yf_saturated_input_data.loc[:max_enthal_index]
    R1234yf_saturated_input_data_9_2 = R1234yf_saturated_input_data.loc[max_enthal_index:]


    filepath='Refrigerants/'+'R1234yf'+'_isotherms_saturation.csv'
    R1234yf_isotherms=pd.read_csv(filepath, encoding='cp1252')
    R1234yf_isotherms['Pressure (Pa)']=R1234yf_isotherms['Pressure (Pa)']/100000

    R1234yf_isotherm_minus30=R1234yf_isotherms[R1234yf_isotherms['Temperature']==-30]
    R1234yf_isotherm_minus10=R1234yf_isotherms[R1234yf_isotherms['Temperature']==-10]
    R1234yf_isotherm_10=R1234yf_isotherms[R1234yf_isotherms['Temperature']==10]
    R1234yf_isotherm_30=R1234yf_isotherms[R1234yf_isotherms['Temperature']==30]
    R1234yf_isotherm_50=R1234yf_isotherms[R1234yf_isotherms['Temperature']==50]
    R1234yf_isotherm_70=R1234yf_isotherms[R1234yf_isotherms['Temperature']==70]
    R1234yf_isotherm_90=R1234yf_isotherms[R1234yf_isotherms['Temperature']==90]
    R1234yf_isotherm_110=R1234yf_isotherms[R1234yf_isotherms['Temperature']==110]
    R1234yf_isotherm_130=R1234yf_isotherms[R1234yf_isotherms['Temperature']==130]


    series_1 = pd.Series(R1234yf_saturated_input_data['ENTHA0'])
    series_2 = pd.Series(R1234yf_saturated_input_data['ENTHA1'])
    series_2=series_2[::-1]

    Enthal = pd.concat([series_1, series_2], ignore_index=True)

    series_a = pd.Series(R1234yf_saturated_input_data['DRUCK'])
    series_b = pd.Series(R1234yf_saturated_input_data['DRUCK'])
    series_b=series_b[::-1]
    DRUCK = pd.concat([series_a, series_b], ignore_index=True)
    concatenated_df = pd.concat([Enthal, DRUCK], axis=1)
    concatenated_df.rename(columns={0:'Enthalpy','DRUCK': 'Pressure'}, inplace=True)


    max_enthal_index = concatenated_df['Enthalpy'].idxmax()
    df1 = concatenated_df.loc[:max_enthal_index]
    df2 = concatenated_df.loc[max_enthal_index:]


    min_enthal_index = R1234yf_isotherm_minus30['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_minus30_1 = R1234yf_isotherm_minus30.loc[:min_enthal_index]
    df_isotherm_minus30_2 = R1234yf_isotherm_minus30.loc[min_enthal_index:]
    min_enthal_index = R1234yf_isotherm_minus10['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_minus10_1 = R1234yf_isotherm_minus10.loc[:min_enthal_index]
    df_isotherm_minus10_2 = R1234yf_isotherm_minus10.loc[min_enthal_index:]
    min_enthal_index = R1234yf_isotherm_10['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_10_1 = R1234yf_isotherm_10.loc[:min_enthal_index]
    df_isotherm_10_2 = R1234yf_isotherm_10.loc[min_enthal_index:]
    min_enthal_index = R1234yf_isotherm_30['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_30_1 = R1234yf_isotherm_30.loc[:min_enthal_index]
    df_isotherm_30_2 = R1234yf_isotherm_30.loc[min_enthal_index:]
    min_enthal_index = R1234yf_isotherm_50['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_50_1 = R1234yf_isotherm_50.loc[:min_enthal_index]
    df_isotherm_50_2 = R1234yf_isotherm_50.loc[min_enthal_index:]
    min_enthal_index = R1234yf_isotherm_70['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_70_1 = R1234yf_isotherm_70.loc[:min_enthal_index]
    df_isotherm_70_2 = R1234yf_isotherm_70.loc[min_enthal_index:]
    min_enthal_index = R1234yf_isotherm_90['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_90_1 = R1234yf_isotherm_90.loc[:min_enthal_index]
    df_isotherm_90_2 = R1234yf_isotherm_90.loc[min_enthal_index:]
    min_enthal_index = R1234yf_isotherm_110['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_110_1 = R1234yf_isotherm_110.loc[:min_enthal_index]
    df_isotherm_110_2 = R1234yf_isotherm_110.loc[min_enthal_index:]
    min_enthal_index = R1234yf_isotherm_130['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_130_1 = R1234yf_isotherm_130.loc[:min_enthal_index]
    df_isotherm_130_2 = R1234yf_isotherm_130.loc[min_enthal_index:]


    fig,ax = plt.subplots(figsize=(10.4,8.1))

    # Initial 4 coordinates (x and y)
    x_coords = np.array([0, 0, 0, 0])
    y_coords = np.array([0, 0, 0, 0])

    # Plot the initial 4 points
    points = ax.scatter(x_coords, y_coords, color='black',s=20)

    # Line connecting the 4 points in sequence (initially)
    line, = ax.plot(x_coords, y_coords, color='blue', linewidth=1)


    sns.lineplot(data=df_isotherm_minus30_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_minus30_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_minus10_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_minus10_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_10_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_10_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_30_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_30_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_50_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_50_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_70_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_70_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_90_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_90_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_110_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_110_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_130_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_130_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')

    sns.lineplot(data=R1234yf_saturated_input_data_1_1, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_2_1, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_3_1, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_4_1, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_5_1, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_6_1, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_7_1, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_8_1, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_9_1, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_1_2, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_2_2, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_3_2, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_4_2, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_5_2, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_6_2, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_7_2, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_8_2, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1234yf_saturated_input_data_9_2, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')

    sns.lineplot(data=df1, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')
    sns.lineplot(data=df2, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')

    # Set plot title and axis labels
    ax.set_title('R1234yf', fontsize=10)
    ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=10)
    ax.set_ylabel('Pressure [barA]', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(100, 550)  # Set x-axis limits
    ax.set_ylim(0.9, 200)  # Set y-axis limits
    ax.tick_params(axis='both', labelsize=10)


    plt.show(block=False)
    return(points,line,fig,ax)

def R1233zd_diag():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np




    filepath= 'Refrigerants/' + "R1233zd.csv"

    R1233zd_saturated_input_data=pd.read_csv(filepath, encoding='cp1252')
    R1233zd_saturated_input_data['DRUCK']=R1233zd_saturated_input_data['DRUCK']/100000
    R1233zd_saturated_input_data['Enthalpy_Quality0.1']=R1233zd_saturated_input_data['ENTHA0']+0.1*(R1233zd_saturated_input_data['ENTHA1']-R1233zd_saturated_input_data['ENTHA0'])
    R1233zd_saturated_input_data['Enthalpy_Quality0.2']=R1233zd_saturated_input_data['ENTHA0']+0.2*(R1233zd_saturated_input_data['ENTHA1']-R1233zd_saturated_input_data['ENTHA0'])
    R1233zd_saturated_input_data['Enthalpy_Quality0.3']=R1233zd_saturated_input_data['ENTHA0']+0.3*(R1233zd_saturated_input_data['ENTHA1']-R1233zd_saturated_input_data['ENTHA0'])
    R1233zd_saturated_input_data['Enthalpy_Quality0.4']=R1233zd_saturated_input_data['ENTHA0']+0.4*(R1233zd_saturated_input_data['ENTHA1']-R1233zd_saturated_input_data['ENTHA0'])
    R1233zd_saturated_input_data['Enthalpy_Quality0.5']=R1233zd_saturated_input_data['ENTHA0']+0.5*(R1233zd_saturated_input_data['ENTHA1']-R1233zd_saturated_input_data['ENTHA0'])
    R1233zd_saturated_input_data['Enthalpy_Quality0.6']=R1233zd_saturated_input_data['ENTHA0']+0.6*(R1233zd_saturated_input_data['ENTHA1']-R1233zd_saturated_input_data['ENTHA0'])
    R1233zd_saturated_input_data['Enthalpy_Quality0.7']=R1233zd_saturated_input_data['ENTHA0']+0.7*(R1233zd_saturated_input_data['ENTHA1']-R1233zd_saturated_input_data['ENTHA0'])
    R1233zd_saturated_input_data['Enthalpy_Quality0.8']=R1233zd_saturated_input_data['ENTHA0']+0.8*(R1233zd_saturated_input_data['ENTHA1']-R1233zd_saturated_input_data['ENTHA0'])
    R1233zd_saturated_input_data['Enthalpy_Quality0.9']=R1233zd_saturated_input_data['ENTHA0']+0.9*(R1233zd_saturated_input_data['ENTHA1']-R1233zd_saturated_input_data['ENTHA0'])

    max_enthal_index = R1233zd_saturated_input_data['Enthalpy_Quality0.1'].idxmax()
    R1233zd_saturated_input_data_1_1 = R1233zd_saturated_input_data.loc[:max_enthal_index]
    R1233zd_saturated_input_data_1_2 = R1233zd_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1233zd_saturated_input_data['Enthalpy_Quality0.2'].idxmax()
    R1233zd_saturated_input_data_2_1 = R1233zd_saturated_input_data.loc[:max_enthal_index]
    R1233zd_saturated_input_data_2_2 = R1233zd_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1233zd_saturated_input_data['Enthalpy_Quality0.3'].idxmax()
    R1233zd_saturated_input_data_3_1 = R1233zd_saturated_input_data.loc[:max_enthal_index]
    R1233zd_saturated_input_data_3_2 = R1233zd_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1233zd_saturated_input_data['Enthalpy_Quality0.4'].idxmax()
    R1233zd_saturated_input_data_4_1 = R1233zd_saturated_input_data.loc[:max_enthal_index]
    R1233zd_saturated_input_data_4_2 = R1233zd_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1233zd_saturated_input_data['Enthalpy_Quality0.5'].idxmax()
    R1233zd_saturated_input_data_5_1 = R1233zd_saturated_input_data.loc[:max_enthal_index]
    R1233zd_saturated_input_data_5_2 = R1233zd_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1233zd_saturated_input_data['Enthalpy_Quality0.6'].idxmax()
    R1233zd_saturated_input_data_6_1 = R1233zd_saturated_input_data.loc[:max_enthal_index]
    R1233zd_saturated_input_data_6_2 = R1233zd_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1233zd_saturated_input_data['Enthalpy_Quality0.7'].idxmax()
    R1233zd_saturated_input_data_7_1 = R1233zd_saturated_input_data.loc[:max_enthal_index]
    R1233zd_saturated_input_data_7_2 = R1233zd_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1233zd_saturated_input_data['Enthalpy_Quality0.8'].idxmax()
    R1233zd_saturated_input_data_8_1 = R1233zd_saturated_input_data.loc[:max_enthal_index]
    R1233zd_saturated_input_data_8_2 = R1233zd_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R1233zd_saturated_input_data['Enthalpy_Quality0.9'].idxmax()
    R1233zd_saturated_input_data_9_1 = R1233zd_saturated_input_data.loc[:max_enthal_index]
    R1233zd_saturated_input_data_9_2 = R1233zd_saturated_input_data.loc[max_enthal_index:]



    series_1 = pd.Series(R1233zd_saturated_input_data['ENTHA0'])
    series_2 = pd.Series(R1233zd_saturated_input_data['ENTHA1'])
    series_2=series_2[::-1]

    Enthal = pd.concat([series_1, series_2], ignore_index=True)

    series_a = pd.Series(R1233zd_saturated_input_data['DRUCK'])
    series_b = pd.Series(R1233zd_saturated_input_data['DRUCK'])
    series_b=series_b[::-1]
    DRUCK = pd.concat([series_a, series_b], ignore_index=True)
    concatenated_df = pd.concat([Enthal, DRUCK], axis=1)
    concatenated_df.rename(columns={0:'Enthalpy','DRUCK': 'Pressure'}, inplace=True)


    max_enthal_index = concatenated_df['Enthalpy'].idxmax()
    df1 = concatenated_df.loc[:max_enthal_index]
    df2 = concatenated_df.loc[max_enthal_index:]

    fig,ax = plt.subplots(figsize=(10.4,8.1))

    # Initial 4 coordinates (x and y)
    x_coords = np.array([0, 0, 0, 0])
    y_coords = np.array([0, 0, 0, 0])

    # Plot the initial 4 points
    points = ax.scatter(x_coords, y_coords, color='black',s=20)

    # Line connecting the 4 points in sequence (initially)
    line, = ax.plot(x_coords, y_coords, color='blue', linewidth=1)

    sns.lineplot(data=R1233zd_saturated_input_data_1_1, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_2_1, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_3_1, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_4_1, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_5_1, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_6_1, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_7_1, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_8_1, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_9_1, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_1_2, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_2_2, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_3_2, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_4_2, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_5_2, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_6_2, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_7_2, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_8_2, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R1233zd_saturated_input_data_9_2, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')

    sns.lineplot(data=df1, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')
    sns.lineplot(data=df2, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')


    # Set plot title and axis labels
    ax.set_title('R1233zd', fontsize=10)
    ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=10)
    ax.set_ylabel('Pressure [barA]', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(100, 550)  # Set x-axis limits
    ax.set_ylim(0.9, 200)  # Set y-axis limits
    ax.tick_params(axis='both', labelsize=10)


    plt.show(block=False)
    return(points,line,fig,ax)

def R32_diag():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np



    filepath= 'Refrigerants/' + "R32.csv"

    R32_saturated_input_data=pd.read_csv(filepath, encoding='cp1252')
    R32_saturated_input_data['DRUCK']=R32_saturated_input_data['DRUCK']/100000
    R32_saturated_input_data['Enthalpy_Quality0.1']=R32_saturated_input_data['ENTHA0']+0.1*(R32_saturated_input_data['ENTHA1']-R32_saturated_input_data['ENTHA0'])
    R32_saturated_input_data['Enthalpy_Quality0.2']=R32_saturated_input_data['ENTHA0']+0.2*(R32_saturated_input_data['ENTHA1']-R32_saturated_input_data['ENTHA0'])
    R32_saturated_input_data['Enthalpy_Quality0.3']=R32_saturated_input_data['ENTHA0']+0.3*(R32_saturated_input_data['ENTHA1']-R32_saturated_input_data['ENTHA0'])
    R32_saturated_input_data['Enthalpy_Quality0.4']=R32_saturated_input_data['ENTHA0']+0.4*(R32_saturated_input_data['ENTHA1']-R32_saturated_input_data['ENTHA0'])
    R32_saturated_input_data['Enthalpy_Quality0.5']=R32_saturated_input_data['ENTHA0']+0.5*(R32_saturated_input_data['ENTHA1']-R32_saturated_input_data['ENTHA0'])
    R32_saturated_input_data['Enthalpy_Quality0.6']=R32_saturated_input_data['ENTHA0']+0.6*(R32_saturated_input_data['ENTHA1']-R32_saturated_input_data['ENTHA0'])
    R32_saturated_input_data['Enthalpy_Quality0.7']=R32_saturated_input_data['ENTHA0']+0.7*(R32_saturated_input_data['ENTHA1']-R32_saturated_input_data['ENTHA0'])
    R32_saturated_input_data['Enthalpy_Quality0.8']=R32_saturated_input_data['ENTHA0']+0.8*(R32_saturated_input_data['ENTHA1']-R32_saturated_input_data['ENTHA0'])
    R32_saturated_input_data['Enthalpy_Quality0.9']=R32_saturated_input_data['ENTHA0']+0.9*(R32_saturated_input_data['ENTHA1']-R32_saturated_input_data['ENTHA0'])

    max_enthal_index = R32_saturated_input_data['Enthalpy_Quality0.1'].idxmax()
    R32_saturated_input_data_1_1 = R32_saturated_input_data.loc[:max_enthal_index]
    R32_saturated_input_data_1_2 = R32_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R32_saturated_input_data['Enthalpy_Quality0.2'].idxmax()
    R32_saturated_input_data_2_1 = R32_saturated_input_data.loc[:max_enthal_index]
    R32_saturated_input_data_2_2 = R32_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R32_saturated_input_data['Enthalpy_Quality0.3'].idxmax()
    R32_saturated_input_data_3_1 = R32_saturated_input_data.loc[:max_enthal_index]
    R32_saturated_input_data_3_2 = R32_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R32_saturated_input_data['Enthalpy_Quality0.4'].idxmax()
    R32_saturated_input_data_4_1 = R32_saturated_input_data.loc[:max_enthal_index]
    R32_saturated_input_data_4_2 = R32_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R32_saturated_input_data['Enthalpy_Quality0.5'].idxmax()
    R32_saturated_input_data_5_1 = R32_saturated_input_data.loc[:max_enthal_index]
    R32_saturated_input_data_5_2 = R32_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R32_saturated_input_data['Enthalpy_Quality0.6'].idxmax()
    R32_saturated_input_data_6_1 = R32_saturated_input_data.loc[:max_enthal_index]
    R32_saturated_input_data_6_2 = R32_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R32_saturated_input_data['Enthalpy_Quality0.7'].idxmax()
    R32_saturated_input_data_7_1 = R32_saturated_input_data.loc[:max_enthal_index]
    R32_saturated_input_data_7_2 = R32_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R32_saturated_input_data['Enthalpy_Quality0.8'].idxmax()
    R32_saturated_input_data_8_1 = R32_saturated_input_data.loc[:max_enthal_index]
    R32_saturated_input_data_8_2 = R32_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R32_saturated_input_data['Enthalpy_Quality0.9'].idxmax()
    R32_saturated_input_data_9_1 = R32_saturated_input_data.loc[:max_enthal_index]
    R32_saturated_input_data_9_2 = R32_saturated_input_data.loc[max_enthal_index:]



    series_1 = pd.Series(R32_saturated_input_data['ENTHA0'])
    series_2 = pd.Series(R32_saturated_input_data['ENTHA1'])
    series_2=series_2[::-1]

    Enthal = pd.concat([series_1, series_2], ignore_index=True)

    series_a = pd.Series(R32_saturated_input_data['DRUCK'])
    series_b = pd.Series(R32_saturated_input_data['DRUCK'])
    series_b=series_b[::-1]
    DRUCK = pd.concat([series_a, series_b], ignore_index=True)
    concatenated_df = pd.concat([Enthal, DRUCK], axis=1)
    concatenated_df.rename(columns={0:'Enthalpy','DRUCK': 'Pressure'}, inplace=True)


    max_enthal_index = concatenated_df['Enthalpy'].idxmax()
    df1 = concatenated_df.loc[:max_enthal_index]
    df2 = concatenated_df.loc[max_enthal_index:]

    fig,ax = plt.subplots(figsize=(10.4,8.1))
    
    # Initial 4 coordinates (x and y)
    x_coords = np.array([0, 0, 0, 0])
    y_coords = np.array([0, 0, 0, 0])

    # Plot the initial 4 points
    points = ax.scatter(x_coords, y_coords, color='black',s=20)

    # Line connecting the 4 points in sequence (initially)
    line, = ax.plot(x_coords, y_coords, color='blue', linewidth=1)

    sns.lineplot(data=R32_saturated_input_data_1_1, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_2_1, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_3_1, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_4_1, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_5_1, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_6_1, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_7_1, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_8_1, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_9_1, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_1_2, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_2_2, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_3_2, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_4_2, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_5_2, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_6_2, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_7_2, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_8_2, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R32_saturated_input_data_9_2, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')

    sns.lineplot(data=df1, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')
    sns.lineplot(data=df2, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')



    # Set plot title and axis labels
    ax.set_title('R32', fontsize=10)
    ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=10)
    ax.set_ylabel('Pressure [barA]', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(100, 650)  # Set x-axis limits
    ax.set_ylim(0.9, 200)  # Set y-axis limits
    ax.tick_params(axis='both', labelsize=10)


    plt.show(block=False)
    return(points,line,fig,ax)

def R290_diag():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    filepath= 'Refrigerants/' + "R290.csv"

    R290_saturated_input_data=pd.read_csv(filepath, encoding='cp1252')
    R290_saturated_input_data['DRUCK']=R290_saturated_input_data['DRUCK']/100000
    R290_saturated_input_data['Enthalpy_Quality0.1']=R290_saturated_input_data['ENTHA0']+0.1*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.2']=R290_saturated_input_data['ENTHA0']+0.2*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.3']=R290_saturated_input_data['ENTHA0']+0.3*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.4']=R290_saturated_input_data['ENTHA0']+0.4*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.5']=R290_saturated_input_data['ENTHA0']+0.5*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.6']=R290_saturated_input_data['ENTHA0']+0.6*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.7']=R290_saturated_input_data['ENTHA0']+0.7*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.8']=R290_saturated_input_data['ENTHA0']+0.8*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.9']=R290_saturated_input_data['ENTHA0']+0.9*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])

    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.1'].idxmax()
    R290_saturated_input_data_1_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_1_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.2'].idxmax()
    R290_saturated_input_data_2_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_2_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.3'].idxmax()
    R290_saturated_input_data_3_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_3_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.4'].idxmax()
    R290_saturated_input_data_4_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_4_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.5'].idxmax()
    R290_saturated_input_data_5_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_5_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.6'].idxmax()
    R290_saturated_input_data_6_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_6_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.7'].idxmax()
    R290_saturated_input_data_7_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_7_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.8'].idxmax()
    R290_saturated_input_data_8_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_8_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.9'].idxmax()
    R290_saturated_input_data_9_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_9_2 = R290_saturated_input_data.loc[max_enthal_index:]



    series_1 = pd.Series(R290_saturated_input_data['ENTHA0'])
    series_2 = pd.Series(R290_saturated_input_data['ENTHA1'])
    series_2=series_2[::-1]

    Enthal = pd.concat([series_1, series_2], ignore_index=True)

    series_a = pd.Series(R290_saturated_input_data['DRUCK'])
    series_b = pd.Series(R290_saturated_input_data['DRUCK'])
    series_b=series_b[::-1]
    DRUCK = pd.concat([series_a, series_b], ignore_index=True)
    concatenated_df = pd.concat([Enthal, DRUCK], axis=1)
    concatenated_df.rename(columns={0:'Enthalpy','DRUCK': 'Pressure'}, inplace=True)


    max_enthal_index = concatenated_df['Enthalpy'].idxmax()
    df1 = concatenated_df.loc[:max_enthal_index]
    df2 = concatenated_df.loc[max_enthal_index:]

    fig,ax = plt.subplots(figsize=(10.4,8.1))
    
    # Initial 4 coordinates (x and y)
    x_coords = np.array([0, 0, 0, 0])
    y_coords = np.array([0, 0, 0, 0])

    # Plot the initial 4 points
    points = ax.scatter(x_coords, y_coords, color='black',s=20)

    # Line connecting the 4 points in sequence (initially)
    line, = ax.plot(x_coords, y_coords, color='blue', linewidth=1)

    sns.lineplot(data=R290_saturated_input_data_1_1, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_2_1, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_3_1, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_4_1, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_5_1, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_6_1, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_7_1, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_8_1, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_9_1, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_1_2, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_2_2, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_3_2, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_4_2, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_5_2, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_6_2, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_7_2, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_8_2, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_9_2, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')

    sns.lineplot(data=df1, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')
    sns.lineplot(data=df2, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')



    # Set plot title and axis labels
    ax.set_title('R290', fontsize=10)
    ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=10)
    ax.set_ylabel('Pressure [barA]', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(100, 550)  # Set x-axis limits
    ax.set_ylim(0.9, 200)  # Set y-axis limits
    ax.tick_params(axis='both', labelsize=10)


    plt.show(block=False)
    return(points,line,fig,ax)

def R12_diag():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np


    filepath= 'Refrigerants/' + "R12.csv"
 

    R12_saturated_input_data=pd.read_csv(filepath, encoding='cp1252')
    R12_saturated_input_data['DRUCK']=R12_saturated_input_data['DRUCK']/100000
    R12_saturated_input_data['Enthalpy_Quality0.1']=R12_saturated_input_data['ENTHA0']+0.1*(R12_saturated_input_data['ENTHA1']-R12_saturated_input_data['ENTHA0'])
    R12_saturated_input_data['Enthalpy_Quality0.2']=R12_saturated_input_data['ENTHA0']+0.2*(R12_saturated_input_data['ENTHA1']-R12_saturated_input_data['ENTHA0'])
    R12_saturated_input_data['Enthalpy_Quality0.3']=R12_saturated_input_data['ENTHA0']+0.3*(R12_saturated_input_data['ENTHA1']-R12_saturated_input_data['ENTHA0'])
    R12_saturated_input_data['Enthalpy_Quality0.4']=R12_saturated_input_data['ENTHA0']+0.4*(R12_saturated_input_data['ENTHA1']-R12_saturated_input_data['ENTHA0'])
    R12_saturated_input_data['Enthalpy_Quality0.5']=R12_saturated_input_data['ENTHA0']+0.5*(R12_saturated_input_data['ENTHA1']-R12_saturated_input_data['ENTHA0'])
    R12_saturated_input_data['Enthalpy_Quality0.6']=R12_saturated_input_data['ENTHA0']+0.6*(R12_saturated_input_data['ENTHA1']-R12_saturated_input_data['ENTHA0'])
    R12_saturated_input_data['Enthalpy_Quality0.7']=R12_saturated_input_data['ENTHA0']+0.7*(R12_saturated_input_data['ENTHA1']-R12_saturated_input_data['ENTHA0'])
    R12_saturated_input_data['Enthalpy_Quality0.8']=R12_saturated_input_data['ENTHA0']+0.8*(R12_saturated_input_data['ENTHA1']-R12_saturated_input_data['ENTHA0'])
    R12_saturated_input_data['Enthalpy_Quality0.9']=R12_saturated_input_data['ENTHA0']+0.9*(R12_saturated_input_data['ENTHA1']-R12_saturated_input_data['ENTHA0'])

    max_enthal_index = R12_saturated_input_data['Enthalpy_Quality0.1'].idxmax()
    R12_saturated_input_data_1_1 = R12_saturated_input_data.loc[:max_enthal_index]
    R12_saturated_input_data_1_2 = R12_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R12_saturated_input_data['Enthalpy_Quality0.2'].idxmax()
    R12_saturated_input_data_2_1 = R12_saturated_input_data.loc[:max_enthal_index]
    R12_saturated_input_data_2_2 = R12_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R12_saturated_input_data['Enthalpy_Quality0.3'].idxmax()
    R12_saturated_input_data_3_1 = R12_saturated_input_data.loc[:max_enthal_index]
    R12_saturated_input_data_3_2 = R12_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R12_saturated_input_data['Enthalpy_Quality0.4'].idxmax()
    R12_saturated_input_data_4_1 = R12_saturated_input_data.loc[:max_enthal_index]
    R12_saturated_input_data_4_2 = R12_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R12_saturated_input_data['Enthalpy_Quality0.5'].idxmax()
    R12_saturated_input_data_5_1 = R12_saturated_input_data.loc[:max_enthal_index]
    R12_saturated_input_data_5_2 = R12_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R12_saturated_input_data['Enthalpy_Quality0.6'].idxmax()
    R12_saturated_input_data_6_1 = R12_saturated_input_data.loc[:max_enthal_index]
    R12_saturated_input_data_6_2 = R12_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R12_saturated_input_data['Enthalpy_Quality0.7'].idxmax()
    R12_saturated_input_data_7_1 = R12_saturated_input_data.loc[:max_enthal_index]
    R12_saturated_input_data_7_2 = R12_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R12_saturated_input_data['Enthalpy_Quality0.8'].idxmax()
    R12_saturated_input_data_8_1 = R12_saturated_input_data.loc[:max_enthal_index]
    R12_saturated_input_data_8_2 = R12_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R12_saturated_input_data['Enthalpy_Quality0.9'].idxmax()
    R12_saturated_input_data_9_1 = R12_saturated_input_data.loc[:max_enthal_index]
    R12_saturated_input_data_9_2 = R12_saturated_input_data.loc[max_enthal_index:]



    series_1 = pd.Series(R12_saturated_input_data['ENTHA0'])
    series_2 = pd.Series(R12_saturated_input_data['ENTHA1'])
    series_2=series_2[::-1]

    Enthal = pd.concat([series_1, series_2], ignore_index=True)

    series_a = pd.Series(R12_saturated_input_data['DRUCK'])
    series_b = pd.Series(R12_saturated_input_data['DRUCK'])
    series_b=series_b[::-1]
    DRUCK = pd.concat([series_a, series_b], ignore_index=True)
    concatenated_df = pd.concat([Enthal, DRUCK], axis=1)
    concatenated_df.rename(columns={0:'Enthalpy','DRUCK': 'Pressure'}, inplace=True)


    max_enthal_index = concatenated_df['Enthalpy'].idxmax()
    df1 = concatenated_df.loc[:max_enthal_index]
    df2 = concatenated_df.loc[max_enthal_index:]


    fig,ax = plt.subplots(figsize=(10.4,8.1))
    
    # Initial 4 coordinates (x and y)
    x_coords = np.array([0, 0, 0, 0])
    y_coords = np.array([0, 0, 0, 0])

    # Plot the initial 4 points
    points = ax.scatter(x_coords, y_coords, color='black',s=20)

    # Line connecting the 4 points in sequence (initially)
    line, = ax.plot(x_coords, y_coords, color='blue', linewidth=1)

    sns.lineplot(data=R12_saturated_input_data_1_1, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_2_1, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_3_1, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_4_1, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_5_1, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_6_1, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_7_1, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_8_1, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_9_1, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_1_2, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_2_2, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_3_2, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_4_2, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_5_2, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_6_2, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_7_2, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_8_2, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R12_saturated_input_data_9_2, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')

    sns.lineplot(data=df1, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')
    sns.lineplot(data=df2, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')




    # Set plot title and axis labels
    ax.set_title('R12', fontsize=10)
    ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=10)
    ax.set_ylabel('Pressure [barA]', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(100, 450)  # Set x-axis limits
    ax.set_ylim(0.9, 200)  # Set y-axis limits
    ax.tick_params(axis='both', labelsize=10)


    plt.show(block=False)
    return(points,line,fig,ax)

def R152A_diag():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    import time


    filepath= 'Refrigerants/' + "R152A.csv"

    R152A_saturated_input_data=pd.read_csv(filepath, encoding='cp1252')
    R152A_saturated_input_data['DRUCK']=R152A_saturated_input_data['DRUCK']/100000
    R152A_saturated_input_data['Enthalpy_Quality0.1']=R152A_saturated_input_data['ENTHA0']+0.1*(R152A_saturated_input_data['ENTHA1']-R152A_saturated_input_data['ENTHA0'])
    R152A_saturated_input_data['Enthalpy_Quality0.2']=R152A_saturated_input_data['ENTHA0']+0.2*(R152A_saturated_input_data['ENTHA1']-R152A_saturated_input_data['ENTHA0'])
    R152A_saturated_input_data['Enthalpy_Quality0.3']=R152A_saturated_input_data['ENTHA0']+0.3*(R152A_saturated_input_data['ENTHA1']-R152A_saturated_input_data['ENTHA0'])
    R152A_saturated_input_data['Enthalpy_Quality0.4']=R152A_saturated_input_data['ENTHA0']+0.4*(R152A_saturated_input_data['ENTHA1']-R152A_saturated_input_data['ENTHA0'])
    R152A_saturated_input_data['Enthalpy_Quality0.5']=R152A_saturated_input_data['ENTHA0']+0.5*(R152A_saturated_input_data['ENTHA1']-R152A_saturated_input_data['ENTHA0'])
    R152A_saturated_input_data['Enthalpy_Quality0.6']=R152A_saturated_input_data['ENTHA0']+0.6*(R152A_saturated_input_data['ENTHA1']-R152A_saturated_input_data['ENTHA0'])
    R152A_saturated_input_data['Enthalpy_Quality0.7']=R152A_saturated_input_data['ENTHA0']+0.7*(R152A_saturated_input_data['ENTHA1']-R152A_saturated_input_data['ENTHA0'])
    R152A_saturated_input_data['Enthalpy_Quality0.8']=R152A_saturated_input_data['ENTHA0']+0.8*(R152A_saturated_input_data['ENTHA1']-R152A_saturated_input_data['ENTHA0'])
    R152A_saturated_input_data['Enthalpy_Quality0.9']=R152A_saturated_input_data['ENTHA0']+0.9*(R152A_saturated_input_data['ENTHA1']-R152A_saturated_input_data['ENTHA0'])

    max_enthal_index = R152A_saturated_input_data['Enthalpy_Quality0.1'].idxmax()
    R152A_saturated_input_data_1_1 = R152A_saturated_input_data.loc[:max_enthal_index]
    R152A_saturated_input_data_1_2 = R152A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R152A_saturated_input_data['Enthalpy_Quality0.2'].idxmax()
    R152A_saturated_input_data_2_1 = R152A_saturated_input_data.loc[:max_enthal_index]
    R152A_saturated_input_data_2_2 = R152A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R152A_saturated_input_data['Enthalpy_Quality0.3'].idxmax()
    R152A_saturated_input_data_3_1 = R152A_saturated_input_data.loc[:max_enthal_index]
    R152A_saturated_input_data_3_2 = R152A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R152A_saturated_input_data['Enthalpy_Quality0.4'].idxmax()
    R152A_saturated_input_data_4_1 = R152A_saturated_input_data.loc[:max_enthal_index]
    R152A_saturated_input_data_4_2 = R152A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R152A_saturated_input_data['Enthalpy_Quality0.5'].idxmax()
    R152A_saturated_input_data_5_1 = R152A_saturated_input_data.loc[:max_enthal_index]
    R152A_saturated_input_data_5_2 = R152A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R152A_saturated_input_data['Enthalpy_Quality0.6'].idxmax()
    R152A_saturated_input_data_6_1 = R152A_saturated_input_data.loc[:max_enthal_index]
    R152A_saturated_input_data_6_2 = R152A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R152A_saturated_input_data['Enthalpy_Quality0.7'].idxmax()
    R152A_saturated_input_data_7_1 = R152A_saturated_input_data.loc[:max_enthal_index]
    R152A_saturated_input_data_7_2 = R152A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R152A_saturated_input_data['Enthalpy_Quality0.8'].idxmax()
    R152A_saturated_input_data_8_1 = R152A_saturated_input_data.loc[:max_enthal_index]
    R152A_saturated_input_data_8_2 = R152A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R152A_saturated_input_data['Enthalpy_Quality0.9'].idxmax()
    R152A_saturated_input_data_9_1 = R152A_saturated_input_data.loc[:max_enthal_index]
    R152A_saturated_input_data_9_2 = R152A_saturated_input_data.loc[max_enthal_index:]


    filepath='Refrigerants/'+'R152A'+'_isotherms_saturation.csv'
    R152A_isotherms=pd.read_csv(filepath, encoding='cp1252')
    R152A_isotherms['Pressure (Pa)']=R152A_isotherms['Pressure (Pa)']/100000

    R152A_isotherm_minus30=R152A_isotherms[R152A_isotherms['Temperature']==-30]
    R152A_isotherm_minus10=R152A_isotherms[R152A_isotherms['Temperature']==-10]
    R152A_isotherm_10=R152A_isotherms[R152A_isotherms['Temperature']==10]
    R152A_isotherm_30=R152A_isotherms[R152A_isotherms['Temperature']==30]
    R152A_isotherm_50=R152A_isotherms[R152A_isotherms['Temperature']==50]
    R152A_isotherm_70=R152A_isotherms[R152A_isotherms['Temperature']==70]
    R152A_isotherm_90=R152A_isotherms[R152A_isotherms['Temperature']==90]
    R152A_isotherm_110=R152A_isotherms[R152A_isotherms['Temperature']==110]
    R152A_isotherm_130=R152A_isotherms[R152A_isotherms['Temperature']==130]


    series_1 = pd.Series(R152A_saturated_input_data['ENTHA0'])
    series_2 = pd.Series(R152A_saturated_input_data['ENTHA1'])
    series_2=series_2[::-1]

    Enthal = pd.concat([series_1, series_2], ignore_index=True)

    series_a = pd.Series(R152A_saturated_input_data['DRUCK'])
    series_b = pd.Series(R152A_saturated_input_data['DRUCK'])
    series_b=series_b[::-1]
    DRUCK = pd.concat([series_a, series_b], ignore_index=True)
    concatenated_df = pd.concat([Enthal, DRUCK], axis=1)
    concatenated_df.rename(columns={0:'Enthalpy','DRUCK': 'Pressure'}, inplace=True)


    max_enthal_index = concatenated_df['Enthalpy'].idxmax()
    df1 = concatenated_df.loc[:max_enthal_index]
    df2 = concatenated_df.loc[max_enthal_index:]


    min_enthal_index = R152A_isotherm_minus30['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_minus30_1 = R152A_isotherm_minus30.loc[:min_enthal_index]
    df_isotherm_minus30_2 = R152A_isotherm_minus30.loc[min_enthal_index:]
    min_enthal_index = R152A_isotherm_minus10['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_minus10_1 = R152A_isotherm_minus10.loc[:min_enthal_index]
    df_isotherm_minus10_2 = R152A_isotherm_minus10.loc[min_enthal_index:]
    min_enthal_index = R152A_isotherm_10['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_10_1 = R152A_isotherm_10.loc[:min_enthal_index]
    df_isotherm_10_2 = R152A_isotherm_10.loc[min_enthal_index:]
    min_enthal_index = R152A_isotherm_30['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_30_1 = R152A_isotherm_30.loc[:min_enthal_index]
    df_isotherm_30_2 = R152A_isotherm_30.loc[min_enthal_index:]
    min_enthal_index = R152A_isotherm_50['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_50_1 = R152A_isotherm_50.loc[:min_enthal_index]
    df_isotherm_50_2 = R152A_isotherm_50.loc[min_enthal_index:]
    min_enthal_index = R152A_isotherm_70['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_70_1 = R152A_isotherm_70.loc[:min_enthal_index]
    df_isotherm_70_2 = R152A_isotherm_70.loc[min_enthal_index:]
    min_enthal_index = R152A_isotherm_90['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_90_1 = R152A_isotherm_90.loc[:min_enthal_index]
    df_isotherm_90_2 = R152A_isotherm_90.loc[min_enthal_index:]
    min_enthal_index = R152A_isotherm_110['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_110_1 = R152A_isotherm_110.loc[:min_enthal_index]
    df_isotherm_110_2 = R152A_isotherm_110.loc[min_enthal_index:]
    min_enthal_index = R152A_isotherm_130['Enthalpy(kJ/kg) '].idxmin()
    df_isotherm_130_1 = R152A_isotherm_130.loc[:min_enthal_index]
    df_isotherm_130_2 = R152A_isotherm_130.loc[min_enthal_index:]


    fig,ax = plt.subplots(figsize=(10.4,8.1))

    # Initial 4 coordinates (x and y)
    x_coords = np.array([0, 0, 0, 0])
    y_coords = np.array([0, 0, 0, 0])

    # Plot the initial 4 points
    points = ax.scatter(x_coords, y_coords, color='black',s=20)

    # Line connecting the 4 points in sequence (initially)
    line, = ax.plot(x_coords, y_coords, color='blue', linewidth=1)


    sns.lineplot(data=df_isotherm_minus30_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_minus30_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_minus10_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_minus10_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_10_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_10_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_30_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_30_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_50_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_50_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_70_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_70_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_90_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_90_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_110_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_110_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_130_1 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')
    sns.lineplot(data=df_isotherm_130_2 , x='Enthalpy(kJ/kg) ', y='Pressure (Pa)',linewidth=0.2,color='red')

    sns.lineplot(data=R152A_saturated_input_data_1_1, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_2_1, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_3_1, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_4_1, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_5_1, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_6_1, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_7_1, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_8_1, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_9_1, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_1_2, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_2_2, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_3_2, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_4_2, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_5_2, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_6_2, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_7_2, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_8_2, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R152A_saturated_input_data_9_2, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')

    sns.lineplot(data=df1, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')
    sns.lineplot(data=df2, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')

    # Set plot title and axis labels
    ax.set_title('R152A', fontsize=10)
    ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=10)
    ax.set_ylabel('Pressure [barA]', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(100, 700)  # Set x-axis limits
    ax.set_ylim(0.9, 200)  # Set y-axis limits
    ax.tick_params(axis='both', labelsize=10)


    plt.show(block=False)
    return(points,line,fig,ax)

def R410A_diag():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np


    filepath= 'Refrigerants/' + "R410A.csv"
 

    R410A_saturated_input_data=pd.read_csv(filepath, encoding='cp1252')
    R410A_saturated_input_data['DRUCK']=R410A_saturated_input_data['DRUCK']/100000
    R410A_saturated_input_data['Enthalpy_Quality0.1']=R410A_saturated_input_data['ENTHA0']+0.1*(R410A_saturated_input_data['ENTHA1']-R410A_saturated_input_data['ENTHA0'])
    R410A_saturated_input_data['Enthalpy_Quality0.2']=R410A_saturated_input_data['ENTHA0']+0.2*(R410A_saturated_input_data['ENTHA1']-R410A_saturated_input_data['ENTHA0'])
    R410A_saturated_input_data['Enthalpy_Quality0.3']=R410A_saturated_input_data['ENTHA0']+0.3*(R410A_saturated_input_data['ENTHA1']-R410A_saturated_input_data['ENTHA0'])
    R410A_saturated_input_data['Enthalpy_Quality0.4']=R410A_saturated_input_data['ENTHA0']+0.4*(R410A_saturated_input_data['ENTHA1']-R410A_saturated_input_data['ENTHA0'])
    R410A_saturated_input_data['Enthalpy_Quality0.5']=R410A_saturated_input_data['ENTHA0']+0.5*(R410A_saturated_input_data['ENTHA1']-R410A_saturated_input_data['ENTHA0'])
    R410A_saturated_input_data['Enthalpy_Quality0.6']=R410A_saturated_input_data['ENTHA0']+0.6*(R410A_saturated_input_data['ENTHA1']-R410A_saturated_input_data['ENTHA0'])
    R410A_saturated_input_data['Enthalpy_Quality0.7']=R410A_saturated_input_data['ENTHA0']+0.7*(R410A_saturated_input_data['ENTHA1']-R410A_saturated_input_data['ENTHA0'])
    R410A_saturated_input_data['Enthalpy_Quality0.8']=R410A_saturated_input_data['ENTHA0']+0.8*(R410A_saturated_input_data['ENTHA1']-R410A_saturated_input_data['ENTHA0'])
    R410A_saturated_input_data['Enthalpy_Quality0.9']=R410A_saturated_input_data['ENTHA0']+0.9*(R410A_saturated_input_data['ENTHA1']-R410A_saturated_input_data['ENTHA0'])

    max_enthal_index = R410A_saturated_input_data['Enthalpy_Quality0.1'].idxmax()
    R410A_saturated_input_data_1_1 = R410A_saturated_input_data.loc[:max_enthal_index]
    R410A_saturated_input_data_1_2 = R410A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R410A_saturated_input_data['Enthalpy_Quality0.2'].idxmax()
    R410A_saturated_input_data_2_1 = R410A_saturated_input_data.loc[:max_enthal_index]
    R410A_saturated_input_data_2_2 = R410A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R410A_saturated_input_data['Enthalpy_Quality0.3'].idxmax()
    R410A_saturated_input_data_3_1 = R410A_saturated_input_data.loc[:max_enthal_index]
    R410A_saturated_input_data_3_2 = R410A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R410A_saturated_input_data['Enthalpy_Quality0.4'].idxmax()
    R410A_saturated_input_data_4_1 = R410A_saturated_input_data.loc[:max_enthal_index]
    R410A_saturated_input_data_4_2 = R410A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R410A_saturated_input_data['Enthalpy_Quality0.5'].idxmax()
    R410A_saturated_input_data_5_1 = R410A_saturated_input_data.loc[:max_enthal_index]
    R410A_saturated_input_data_5_2 = R410A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R410A_saturated_input_data['Enthalpy_Quality0.6'].idxmax()
    R410A_saturated_input_data_6_1 = R410A_saturated_input_data.loc[:max_enthal_index]
    R410A_saturated_input_data_6_2 = R410A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R410A_saturated_input_data['Enthalpy_Quality0.7'].idxmax()
    R410A_saturated_input_data_7_1 = R410A_saturated_input_data.loc[:max_enthal_index]
    R410A_saturated_input_data_7_2 = R410A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R410A_saturated_input_data['Enthalpy_Quality0.8'].idxmax()
    R410A_saturated_input_data_8_1 = R410A_saturated_input_data.loc[:max_enthal_index]
    R410A_saturated_input_data_8_2 = R410A_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R410A_saturated_input_data['Enthalpy_Quality0.9'].idxmax()
    R410A_saturated_input_data_9_1 = R410A_saturated_input_data.loc[:max_enthal_index]
    R410A_saturated_input_data_9_2 = R410A_saturated_input_data.loc[max_enthal_index:]



    series_1 = pd.Series(R410A_saturated_input_data['ENTHA0'])
    series_2 = pd.Series(R410A_saturated_input_data['ENTHA1'])
    series_2=series_2[::-1]

    Enthal = pd.concat([series_1, series_2], ignore_index=True)

    series_a = pd.Series(R410A_saturated_input_data['DRUCK'])
    series_b = pd.Series(R410A_saturated_input_data['DRUCK'])
    series_b=series_b[::-1]
    DRUCK = pd.concat([series_a, series_b], ignore_index=True)
    concatenated_df = pd.concat([Enthal, DRUCK], axis=1)
    concatenated_df.rename(columns={0:'Enthalpy','DRUCK': 'Pressure'}, inplace=True)


    max_enthal_index = concatenated_df['Enthalpy'].idxmax()
    df1 = concatenated_df.loc[:max_enthal_index]
    df2 = concatenated_df.loc[max_enthal_index:]


    fig,ax = plt.subplots(figsize=(10.4,8.1))
    
    # Initial 4 coordinates (x and y)
    x_coords = np.array([0, 0, 0, 0])
    y_coords = np.array([0, 0, 0, 0])

    # Plot the initial 4 points
    points = ax.scatter(x_coords, y_coords, color='black',s=20)

    # Line connecting the 4 points in sequence (initially)
    line, = ax.plot(x_coords, y_coords, color='blue', linewidth=1)

    sns.lineplot(data=R410A_saturated_input_data_1_1, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_2_1, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_3_1, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_4_1, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_5_1, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_6_1, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_7_1, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_8_1, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_9_1, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_1_2, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_2_2, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_3_2, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_4_2, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_5_2, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_6_2, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_7_2, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_8_2, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R410A_saturated_input_data_9_2, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')

    sns.lineplot(data=df1, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')
    sns.lineplot(data=df2, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')




    # Set plot title and axis labels
    ax.set_title('R410A', fontsize=10)
    ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=10)
    ax.set_ylabel('Pressure [barA]', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(100, 450)  # Set x-axis limits
    ax.set_ylim(0.9, 200)  # Set y-axis limits
    ax.tick_params(axis='both', labelsize=10)


    plt.show(block=False)
    return(points,line,fig,ax)

def R717_diag():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np


    filepath= 'Refrigerants/' + "R717.csv"
 

    R717_saturated_input_data=pd.read_csv(filepath, encoding='cp1252')
    R717_saturated_input_data['DRUCK']=R717_saturated_input_data['DRUCK']/100000
    R717_saturated_input_data['Enthalpy_Quality0.1']=R717_saturated_input_data['ENTHA0']+0.1*(R717_saturated_input_data['ENTHA1']-R717_saturated_input_data['ENTHA0'])
    R717_saturated_input_data['Enthalpy_Quality0.2']=R717_saturated_input_data['ENTHA0']+0.2*(R717_saturated_input_data['ENTHA1']-R717_saturated_input_data['ENTHA0'])
    R717_saturated_input_data['Enthalpy_Quality0.3']=R717_saturated_input_data['ENTHA0']+0.3*(R717_saturated_input_data['ENTHA1']-R717_saturated_input_data['ENTHA0'])
    R717_saturated_input_data['Enthalpy_Quality0.4']=R717_saturated_input_data['ENTHA0']+0.4*(R717_saturated_input_data['ENTHA1']-R717_saturated_input_data['ENTHA0'])
    R717_saturated_input_data['Enthalpy_Quality0.5']=R717_saturated_input_data['ENTHA0']+0.5*(R717_saturated_input_data['ENTHA1']-R717_saturated_input_data['ENTHA0'])
    R717_saturated_input_data['Enthalpy_Quality0.6']=R717_saturated_input_data['ENTHA0']+0.6*(R717_saturated_input_data['ENTHA1']-R717_saturated_input_data['ENTHA0'])
    R717_saturated_input_data['Enthalpy_Quality0.7']=R717_saturated_input_data['ENTHA0']+0.7*(R717_saturated_input_data['ENTHA1']-R717_saturated_input_data['ENTHA0'])
    R717_saturated_input_data['Enthalpy_Quality0.8']=R717_saturated_input_data['ENTHA0']+0.8*(R717_saturated_input_data['ENTHA1']-R717_saturated_input_data['ENTHA0'])
    R717_saturated_input_data['Enthalpy_Quality0.9']=R717_saturated_input_data['ENTHA0']+0.9*(R717_saturated_input_data['ENTHA1']-R717_saturated_input_data['ENTHA0'])

    max_enthal_index = R717_saturated_input_data['Enthalpy_Quality0.1'].idxmax()
    R717_saturated_input_data_1_1 = R717_saturated_input_data.loc[:max_enthal_index]
    R717_saturated_input_data_1_2 = R717_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R717_saturated_input_data['Enthalpy_Quality0.2'].idxmax()
    R717_saturated_input_data_2_1 = R717_saturated_input_data.loc[:max_enthal_index]
    R717_saturated_input_data_2_2 = R717_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R717_saturated_input_data['Enthalpy_Quality0.3'].idxmax()
    R717_saturated_input_data_3_1 = R717_saturated_input_data.loc[:max_enthal_index]
    R717_saturated_input_data_3_2 = R717_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R717_saturated_input_data['Enthalpy_Quality0.4'].idxmax()
    R717_saturated_input_data_4_1 = R717_saturated_input_data.loc[:max_enthal_index]
    R717_saturated_input_data_4_2 = R717_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R717_saturated_input_data['Enthalpy_Quality0.5'].idxmax()
    R717_saturated_input_data_5_1 = R717_saturated_input_data.loc[:max_enthal_index]
    R717_saturated_input_data_5_2 = R717_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R717_saturated_input_data['Enthalpy_Quality0.6'].idxmax()
    R717_saturated_input_data_6_1 = R717_saturated_input_data.loc[:max_enthal_index]
    R717_saturated_input_data_6_2 = R717_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R717_saturated_input_data['Enthalpy_Quality0.7'].idxmax()
    R717_saturated_input_data_7_1 = R717_saturated_input_data.loc[:max_enthal_index]
    R717_saturated_input_data_7_2 = R717_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R717_saturated_input_data['Enthalpy_Quality0.8'].idxmax()
    R717_saturated_input_data_8_1 = R717_saturated_input_data.loc[:max_enthal_index]
    R717_saturated_input_data_8_2 = R717_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R717_saturated_input_data['Enthalpy_Quality0.9'].idxmax()
    R717_saturated_input_data_9_1 = R717_saturated_input_data.loc[:max_enthal_index]
    R717_saturated_input_data_9_2 = R717_saturated_input_data.loc[max_enthal_index:]



    series_1 = pd.Series(R717_saturated_input_data['ENTHA0'])
    series_2 = pd.Series(R717_saturated_input_data['ENTHA1'])
    series_2=series_2[::-1]

    Enthal = pd.concat([series_1, series_2], ignore_index=True)

    series_a = pd.Series(R717_saturated_input_data['DRUCK'])
    series_b = pd.Series(R717_saturated_input_data['DRUCK'])
    series_b=series_b[::-1]
    DRUCK = pd.concat([series_a, series_b], ignore_index=True)
    concatenated_df = pd.concat([Enthal, DRUCK], axis=1)
    concatenated_df.rename(columns={0:'Enthalpy','DRUCK': 'Pressure'}, inplace=True)


    max_enthal_index = concatenated_df['Enthalpy'].idxmax()
    df1 = concatenated_df.loc[:max_enthal_index]
    df2 = concatenated_df.loc[max_enthal_index:]


    fig,ax = plt.subplots(figsize=(10.4,8.1))
    
    # Initial 4 coordinates (x and y)
    x_coords = np.array([0, 0, 0, 0])
    y_coords = np.array([0, 0, 0, 0])

    # Plot the initial 4 points
    points = ax.scatter(x_coords, y_coords, color='black',s=20)

    # Line connecting the 4 points in sequence (initially)
    line, = ax.plot(x_coords, y_coords, color='blue', linewidth=1)

    sns.lineplot(data=R717_saturated_input_data_1_1, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_2_1, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_3_1, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_4_1, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_5_1, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_6_1, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_7_1, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_8_1, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_9_1, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_1_2, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_2_2, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_3_2, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_4_2, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_5_2, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_6_2, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_7_2, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_8_2, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R717_saturated_input_data_9_2, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')

    sns.lineplot(data=df1, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')
    sns.lineplot(data=df2, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')




    # Set plot title and axis labels
    ax.set_title('R717', fontsize=10)
    ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=10)
    ax.set_ylabel('Pressure [barA]', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(-100, 2500)  # Set x-axis limits
    ax.set_ylim(0.9, 200)  # Set y-axis limits
    ax.tick_params(axis='both', labelsize=10)


    plt.show(block=False)
    return(points,line,fig,ax)

def R718_diag():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np


    filepath= 'Refrigerants/' + "R718.csv"
 

    R718_saturated_input_data=pd.read_csv(filepath, encoding='cp1252')
    R718_saturated_input_data['DRUCK']=R718_saturated_input_data['DRUCK']/100000
    R718_saturated_input_data['Enthalpy_Quality0.1']=R718_saturated_input_data['ENTHA0']+0.1*(R718_saturated_input_data['ENTHA1']-R718_saturated_input_data['ENTHA0'])
    R718_saturated_input_data['Enthalpy_Quality0.2']=R718_saturated_input_data['ENTHA0']+0.2*(R718_saturated_input_data['ENTHA1']-R718_saturated_input_data['ENTHA0'])
    R718_saturated_input_data['Enthalpy_Quality0.3']=R718_saturated_input_data['ENTHA0']+0.3*(R718_saturated_input_data['ENTHA1']-R718_saturated_input_data['ENTHA0'])
    R718_saturated_input_data['Enthalpy_Quality0.4']=R718_saturated_input_data['ENTHA0']+0.4*(R718_saturated_input_data['ENTHA1']-R718_saturated_input_data['ENTHA0'])
    R718_saturated_input_data['Enthalpy_Quality0.5']=R718_saturated_input_data['ENTHA0']+0.5*(R718_saturated_input_data['ENTHA1']-R718_saturated_input_data['ENTHA0'])
    R718_saturated_input_data['Enthalpy_Quality0.6']=R718_saturated_input_data['ENTHA0']+0.6*(R718_saturated_input_data['ENTHA1']-R718_saturated_input_data['ENTHA0'])
    R718_saturated_input_data['Enthalpy_Quality0.7']=R718_saturated_input_data['ENTHA0']+0.7*(R718_saturated_input_data['ENTHA1']-R718_saturated_input_data['ENTHA0'])
    R718_saturated_input_data['Enthalpy_Quality0.8']=R718_saturated_input_data['ENTHA0']+0.8*(R718_saturated_input_data['ENTHA1']-R718_saturated_input_data['ENTHA0'])
    R718_saturated_input_data['Enthalpy_Quality0.9']=R718_saturated_input_data['ENTHA0']+0.9*(R718_saturated_input_data['ENTHA1']-R718_saturated_input_data['ENTHA0'])

    max_enthal_index = R718_saturated_input_data['Enthalpy_Quality0.1'].idxmax()
    R718_saturated_input_data_1_1 = R718_saturated_input_data.loc[:max_enthal_index]
    R718_saturated_input_data_1_2 = R718_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R718_saturated_input_data['Enthalpy_Quality0.2'].idxmax()
    R718_saturated_input_data_2_1 = R718_saturated_input_data.loc[:max_enthal_index]
    R718_saturated_input_data_2_2 = R718_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R718_saturated_input_data['Enthalpy_Quality0.3'].idxmax()
    R718_saturated_input_data_3_1 = R718_saturated_input_data.loc[:max_enthal_index]
    R718_saturated_input_data_3_2 = R718_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R718_saturated_input_data['Enthalpy_Quality0.4'].idxmax()
    R718_saturated_input_data_4_1 = R718_saturated_input_data.loc[:max_enthal_index]
    R718_saturated_input_data_4_2 = R718_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R718_saturated_input_data['Enthalpy_Quality0.5'].idxmax()
    R718_saturated_input_data_5_1 = R718_saturated_input_data.loc[:max_enthal_index]
    R718_saturated_input_data_5_2 = R718_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R718_saturated_input_data['Enthalpy_Quality0.6'].idxmax()
    R718_saturated_input_data_6_1 = R718_saturated_input_data.loc[:max_enthal_index]
    R718_saturated_input_data_6_2 = R718_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R718_saturated_input_data['Enthalpy_Quality0.7'].idxmax()
    R718_saturated_input_data_7_1 = R718_saturated_input_data.loc[:max_enthal_index]
    R718_saturated_input_data_7_2 = R718_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R718_saturated_input_data['Enthalpy_Quality0.8'].idxmax()
    R718_saturated_input_data_8_1 = R718_saturated_input_data.loc[:max_enthal_index]
    R718_saturated_input_data_8_2 = R718_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R718_saturated_input_data['Enthalpy_Quality0.9'].idxmax()
    R718_saturated_input_data_9_1 = R718_saturated_input_data.loc[:max_enthal_index]
    R718_saturated_input_data_9_2 = R718_saturated_input_data.loc[max_enthal_index:]



    series_1 = pd.Series(R718_saturated_input_data['ENTHA0'])
    series_2 = pd.Series(R718_saturated_input_data['ENTHA1'])
    series_2=series_2[::-1]

    Enthal = pd.concat([series_1, series_2], ignore_index=True)

    series_a = pd.Series(R718_saturated_input_data['DRUCK'])
    series_b = pd.Series(R718_saturated_input_data['DRUCK'])
    series_b=series_b[::-1]
    DRUCK = pd.concat([series_a, series_b], ignore_index=True)
    concatenated_df = pd.concat([Enthal, DRUCK], axis=1)
    concatenated_df.rename(columns={0:'Enthalpy','DRUCK': 'Pressure'}, inplace=True)


    max_enthal_index = concatenated_df['Enthalpy'].idxmax()
    df1 = concatenated_df.loc[:max_enthal_index]
    df2 = concatenated_df.loc[max_enthal_index:]


    fig,ax = plt.subplots(figsize=(10.4,8.1))
    
    # Initial 4 coordinates (x and y)
    x_coords = np.array([0, 0, 0, 0])
    y_coords = np.array([0, 0, 0, 0])

    # Plot the initial 4 points
    points = ax.scatter(x_coords, y_coords, color='black',s=20)

    # Line connecting the 4 points in sequence (initially)
    line, = ax.plot(x_coords, y_coords, color='blue', linewidth=1)

    sns.lineplot(data=R718_saturated_input_data_1_1, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_2_1, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_3_1, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_4_1, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_5_1, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_6_1, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_7_1, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_8_1, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_9_1, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_1_2, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_2_2, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_3_2, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_4_2, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_5_2, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_6_2, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_7_2, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_8_2, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R718_saturated_input_data_9_2, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')

    sns.lineplot(data=df1, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')
    sns.lineplot(data=df2, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')




    # Set plot title and axis labels
    ax.set_title('R718', fontsize=10)
    ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=10)
    ax.set_ylabel('Pressure [barA]', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(100, 450)  # Set x-axis limits
    ax.set_ylim(0.9, 200)  # Set y-axis limits
    ax.tick_params(axis='both', labelsize=10)


    plt.show(block=False)
    return(points,line,fig,ax)

def R744_diag():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np


    filepath= 'Refrigerants/' + "R744.csv"
 

    R744_saturated_input_data=pd.read_csv(filepath, encoding='cp1252')
    R744_saturated_input_data['DRUCK']=R744_saturated_input_data['DRUCK']/100000
    R744_saturated_input_data['Enthalpy_Quality0.1']=R744_saturated_input_data['ENTHA0']+0.1*(R744_saturated_input_data['ENTHA1']-R744_saturated_input_data['ENTHA0'])
    R744_saturated_input_data['Enthalpy_Quality0.2']=R744_saturated_input_data['ENTHA0']+0.2*(R744_saturated_input_data['ENTHA1']-R744_saturated_input_data['ENTHA0'])
    R744_saturated_input_data['Enthalpy_Quality0.3']=R744_saturated_input_data['ENTHA0']+0.3*(R744_saturated_input_data['ENTHA1']-R744_saturated_input_data['ENTHA0'])
    R744_saturated_input_data['Enthalpy_Quality0.4']=R744_saturated_input_data['ENTHA0']+0.4*(R744_saturated_input_data['ENTHA1']-R744_saturated_input_data['ENTHA0'])
    R744_saturated_input_data['Enthalpy_Quality0.5']=R744_saturated_input_data['ENTHA0']+0.5*(R744_saturated_input_data['ENTHA1']-R744_saturated_input_data['ENTHA0'])
    R744_saturated_input_data['Enthalpy_Quality0.6']=R744_saturated_input_data['ENTHA0']+0.6*(R744_saturated_input_data['ENTHA1']-R744_saturated_input_data['ENTHA0'])
    R744_saturated_input_data['Enthalpy_Quality0.7']=R744_saturated_input_data['ENTHA0']+0.7*(R744_saturated_input_data['ENTHA1']-R744_saturated_input_data['ENTHA0'])
    R744_saturated_input_data['Enthalpy_Quality0.8']=R744_saturated_input_data['ENTHA0']+0.8*(R744_saturated_input_data['ENTHA1']-R744_saturated_input_data['ENTHA0'])
    R744_saturated_input_data['Enthalpy_Quality0.9']=R744_saturated_input_data['ENTHA0']+0.9*(R744_saturated_input_data['ENTHA1']-R744_saturated_input_data['ENTHA0'])

    max_enthal_index = R744_saturated_input_data['Enthalpy_Quality0.1'].idxmax()
    R744_saturated_input_data_1_1 = R744_saturated_input_data.loc[:max_enthal_index]
    R744_saturated_input_data_1_2 = R744_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R744_saturated_input_data['Enthalpy_Quality0.2'].idxmax()
    R744_saturated_input_data_2_1 = R744_saturated_input_data.loc[:max_enthal_index]
    R744_saturated_input_data_2_2 = R744_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R744_saturated_input_data['Enthalpy_Quality0.3'].idxmax()
    R744_saturated_input_data_3_1 = R744_saturated_input_data.loc[:max_enthal_index]
    R744_saturated_input_data_3_2 = R744_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R744_saturated_input_data['Enthalpy_Quality0.4'].idxmax()
    R744_saturated_input_data_4_1 = R744_saturated_input_data.loc[:max_enthal_index]
    R744_saturated_input_data_4_2 = R744_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R744_saturated_input_data['Enthalpy_Quality0.5'].idxmax()
    R744_saturated_input_data_5_1 = R744_saturated_input_data.loc[:max_enthal_index]
    R744_saturated_input_data_5_2 = R744_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R744_saturated_input_data['Enthalpy_Quality0.6'].idxmax()
    R744_saturated_input_data_6_1 = R744_saturated_input_data.loc[:max_enthal_index]
    R744_saturated_input_data_6_2 = R744_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R744_saturated_input_data['Enthalpy_Quality0.7'].idxmax()
    R744_saturated_input_data_7_1 = R744_saturated_input_data.loc[:max_enthal_index]
    R744_saturated_input_data_7_2 = R744_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R744_saturated_input_data['Enthalpy_Quality0.8'].idxmax()
    R744_saturated_input_data_8_1 = R744_saturated_input_data.loc[:max_enthal_index]
    R744_saturated_input_data_8_2 = R744_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R744_saturated_input_data['Enthalpy_Quality0.9'].idxmax()
    R744_saturated_input_data_9_1 = R744_saturated_input_data.loc[:max_enthal_index]
    R744_saturated_input_data_9_2 = R744_saturated_input_data.loc[max_enthal_index:]



    series_1 = pd.Series(R744_saturated_input_data['ENTHA0'])
    series_2 = pd.Series(R744_saturated_input_data['ENTHA1'])
    series_2=series_2[::-1]

    Enthal = pd.concat([series_1, series_2], ignore_index=True)

    series_a = pd.Series(R744_saturated_input_data['DRUCK'])
    series_b = pd.Series(R744_saturated_input_data['DRUCK'])
    series_b=series_b[::-1]
    DRUCK = pd.concat([series_a, series_b], ignore_index=True)
    concatenated_df = pd.concat([Enthal, DRUCK], axis=1)
    concatenated_df.rename(columns={0:'Enthalpy','DRUCK': 'Pressure'}, inplace=True)


    max_enthal_index = concatenated_df['Enthalpy'].idxmax()
    df1 = concatenated_df.loc[:max_enthal_index]
    df2 = concatenated_df.loc[max_enthal_index:]


    fig,ax = plt.subplots(figsize=(10.4,8.1))
    
    # Initial 4 coordinates (x and y)
    x_coords = np.array([0, 0, 0, 0])
    y_coords = np.array([0, 0, 0, 0])

    # Plot the initial 4 points
    points = ax.scatter(x_coords, y_coords, color='black',s=20)

    # Line connecting the 4 points in sequence (initially)
    line, = ax.plot(x_coords, y_coords, color='blue', linewidth=1)

    sns.lineplot(data=R744_saturated_input_data_1_1, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_2_1, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_3_1, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_4_1, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_5_1, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_6_1, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_7_1, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_8_1, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_9_1, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_1_2, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_2_2, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_3_2, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_4_2, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_5_2, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_6_2, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_7_2, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_8_2, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R744_saturated_input_data_9_2, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')

    sns.lineplot(data=df1, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')
    sns.lineplot(data=df2, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')




    # Set plot title and axis labels
    ax.set_title('R744', fontsize=10)
    ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=10)
    ax.set_ylabel('Pressure [barA]', fontsize=10)
    ax.set_yscale('log')
    ax.set_xlim(100, 450)  # Set x-axis limits
    ax.set_ylim(0.9, 200)  # Set y-axis limits
    ax.tick_params(axis='both', labelsize=10)


    plt.show(block=False)
    return(points,line,fig,ax)