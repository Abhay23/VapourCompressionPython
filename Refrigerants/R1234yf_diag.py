def phdiagram_R1234yf():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns

    filepath= "C:/Abhay/Automotive Thermal management/Python/VapourCompression/Refrigerants/" + "R1234yf.csv"

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


    filepath='C:/Abhay/Automotive Thermal management/Python/VapourCompression/Refrigerants/R1234yf_isotherms_saturation.csv'
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


    fig,ax = plt.subplots(figsize=(3.5,2.7))


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

    #Set plot title and axis labels

    plt.title('R1234yf',fontsize=8)
    plt.xlabel('Enthalpy [kJ/kg]',fontsize=5)
    plt.ylabel('Pressure [barA]',fontsize=5)
    plt.yscale('log') 
    plt.xlim(100, 500)  # Set x-axis limits
    plt.ylim(0.9, 200)  # Set x-axis limits
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    return fig,ax

    # Display the plot
    plt.show()

phdiagram_R1234yf()

