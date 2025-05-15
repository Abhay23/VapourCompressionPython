def ph_diagram():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    import time



    filepath= "C:/Abhay/Automotive Thermal management/Python/VapourCompression/Refrigerants/" + "R134a.csv"

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


    filepath="C:/Abhay/Automotive Thermal management/Python/VapourCompression/Refrigerants/"+'R134a'+'_isotherms_saturation.csv'
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
    x_coords = np.array([300, 400, 300, 400])
    y_coords = np.array([2, 2, 20, 20])

    # Plot the initial 4 points
    points = ax.scatter(x_coords, y_coords, color='black', label='Points')

    # Line connecting the 4 points in sequence (initially)
    line, = ax.plot(x_coords, y_coords, color='blue', linewidth=2, label='Connecting line')



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
    return(points,line)





