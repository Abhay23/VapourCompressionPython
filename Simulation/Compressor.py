def Compressor( Ps,Pd,rho_Compressor_in,h_Compressor_in,entropy_Compressor_in,Compressor_dict,BoundaryConditions_dict):   
    import os
    
    from coolprop import Refr
    import pandas as pd
 
    Pr=Pd/Ps


    filepath = os.getcwd()+'/Inputs/Component_Sheets/' + Compressor_dict["Compressor_efficiency_csv"]     
    Compressor_Table=pd.read_csv(filepath, sep=';')

    P_ratio_values= Compressor_Table['Pressure_ratio'].unique()
    RPM_values= Compressor_Table['Compressor_RPM'].unique()

    def efficiency_calculation(P_ratio_values,RPM_values):
        for i in range(len(P_ratio_values)):
            if P_ratio_values[i]>Pr:
                break

        for j in range(len(RPM_values)):
            if RPM_values[j]>BoundaryConditions_dict["Compressor_RPM"]:
                break

        a1=Compressor_Table[(Compressor_Table['Pressure_ratio'] == P_ratio_values[i-1]) & (Compressor_Table['Compressor_RPM'] ==RPM_values[j-1])]
        a2=Compressor_Table[(Compressor_Table['Pressure_ratio'] == P_ratio_values[i]) & (Compressor_Table['Compressor_RPM'] ==RPM_values[j-1])]
        a3=Compressor_Table[(Compressor_Table['Pressure_ratio'] == P_ratio_values[i-1]) & (Compressor_Table['Compressor_RPM'] ==RPM_values[j])]
        a4=Compressor_Table[(Compressor_Table['Pressure_ratio'] == P_ratio_values[i]) & (Compressor_Table['Compressor_RPM'] ==RPM_values[j])]

        concatenated_df = pd.concat([a1, a2, a3, a4], axis=0)
        concatenated_df = concatenated_df.reset_index(drop=True)

        A1=(concatenated_df.iloc[1,3]-concatenated_df.iloc[0,3])/(concatenated_df.iloc[1,0]-concatenated_df.iloc[0,0])*(Pr-concatenated_df.iloc[0,0])+concatenated_df.iloc[0,3]
        B1=(concatenated_df.iloc[3,3]-concatenated_df.iloc[2,3])/(concatenated_df.iloc[3,0]-concatenated_df.iloc[2,0])*(Pr-concatenated_df.iloc[2,0])+concatenated_df.iloc[2,3]
        C1=(B1-A1)/(concatenated_df.iloc[3,1]-concatenated_df.iloc[0,1])*(BoundaryConditions_dict["Compressor_RPM"]-concatenated_df.iloc[0,1])+A1
        eta_s=C1

        A2=(concatenated_df.iloc[1,2]-concatenated_df.iloc[0,2])/(concatenated_df.iloc[1,0]-concatenated_df.iloc[0,0])*(Pr-concatenated_df.iloc[0,0])+concatenated_df.iloc[0,2]
        B2=(concatenated_df.iloc[3,2]-concatenated_df.iloc[2,2])/(concatenated_df.iloc[3,0]-concatenated_df.iloc[2,0])*(Pr-concatenated_df.iloc[2,0])+concatenated_df.iloc[2,2]
        C2=(B2-A2)/(concatenated_df.iloc[3,1]-concatenated_df.iloc[0,1])*(BoundaryConditions_dict["Compressor_RPM"]-concatenated_df.iloc[0,1])+A2
        eta_v=C2

        return eta_s,eta_v
    

    eta_s,eta_v=efficiency_calculation(P_ratio_values,RPM_values)
    eta_s=min(1,eta_s)
    eta_v=min(1,eta_v)
    
    print(eta_s,eta_v)
    rps=BoundaryConditions_dict["Compressor_RPM"]/60; # Rpm to rps conversion
    Vcc=Compressor_dict["Compressor_CC"]*10**-6; # Displacement of compressor 
    m_dot=rho_Compressor_in*eta_v*rps*Vcc; # Mass flow rate calculation
    print(m_dot)


    Refrigerant,Refrigerant_state,pressure,rho,Tempr,enthalpy,entropy, C_p, lambda_K,viscocity_fluid,x,Superheat_Subcool=Refr(BoundaryConditions_dict["Refrigerant"],8,Pd,entropy_Compressor_in)

    
    h_ideal= enthalpy # enthalpy assuming isentropic compression


    h_real= (h_ideal-h_Compressor_in)/eta_s + h_Compressor_in #actual enthalpy
    h_2=h_real

    print(1111,h_Compressor_in,h_ideal, h_real,eta_s,Pr)
    power_compressor=m_dot*(h_2-h_Compressor_in)
    power_electric_compressor=power_compressor/Compressor_dict["Compressor_eta_m"] 
    print(power_compressor)
    return(m_dot,h_real,power_compressor)