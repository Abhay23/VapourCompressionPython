def Refr(Refrigerant='R134a',input=1,Input1=20,Input2=50):   #-200,100
    if Refrigerant=='R290':
        Refrigerant='Propyne'
    elif Refrigerant=='R1233zd':
        Refrigerant='R1233zd(E)'
    elif Refrigerant=='R717':
        Refrigerant='Ammonia'


    import CoolProp.CoolProp as CP

    if input==1: # temperature and density as input

        rho=float(Input2)  # Input density in kg/m3
        T_actual=273.15+float(Input1) # Input temperature in K


        D = rho            # Density in kg/m^3
        # Calculate Pressure from Temperature and Density
        P= CP.PropsSI('P', 'T', T_actual, 'D', D, Refrigerant)
        T_saturation = CP.PropsSI('T', 'P', P, 'Q', 0, Refrigerant) # Saturation temperature of the liquid
        Superheat_Subcool=round(T_actual - T_saturation , 2)  # Degree of superheating or subcooling 
        enthalpy = CP.PropsSI('H', 'T', T_actual, 'D', D, Refrigerant)/1000   # Enthalpy in kJ/kg
        entropy = CP.PropsSI('S', 'T', T_actual, 'D', D, Refrigerant)/1000   # Entropy in kJ/kg/K
        C_p = CP.PropsSI('C', 'T', T_actual, 'D', D, Refrigerant)        # Specific heat in J/kg/K
        lambda_K = CP.PropsSI('L', 'T', T_actual, 'D', D, Refrigerant)  # Thermal conductivity in W/m/K
        viscosity = CP.PropsSI('V', 'T', T_actual, 'D', D, Refrigerant)*10**6  # Dynamic viscosity in µPa.s
        x= CP.PropsSI('Q', 'T', T_actual, 'D', D, Refrigerant)  # Quality (0 = liquid, 1 = vapor)
        pressure=P/10**5 # Pressure in barA
        Tempr=T_actual-273.15 #Temperatuire in °C

    elif input==2: # Temperature and pressure as input

        P_actual=float(Input2)*10**5 # Input pressure in Pa
        T_actual=273.15+float(Input1) # Input temperature in 

        D = CP.PropsSI('D', 'T', T_actual, 'P', P_actual, Refrigerant)

        enthalpy = CP.PropsSI('H', 'T', T_actual, 'P', P_actual, Refrigerant)/1000  # Enthalpy in kJ/kg
        entropy = CP.PropsSI('S', 'T', T_actual, 'P', P_actual, Refrigerant)/1000   # Entropy in kJ/kg/K
        C_p = CP.PropsSI('C', 'T', T_actual, 'P', P_actual, Refrigerant)        # Specific heat in J/kg/K
        lambda_K = CP.PropsSI('L', 'T', T_actual, 'P', P_actual, Refrigerant)  # Thermal conductivity in W/m/K
        viscosity= CP.PropsSI('V', 'T', T_actual, 'P', P_actual, Refrigerant)*10**6  # Dynamic viscosity in Pa.s
        x = CP.PropsSI('Q', 'T', T_actual, 'P', P_actual, Refrigerant)  # Quality (0 = liquid, 1 = vapor)
        T_saturation = CP.PropsSI('T', 'P', P_actual, 'Q', 0, Refrigerant)  # Saturation temperature of the liquid
        Superheat_Subcool=round(T_actual - T_saturation , 2)  # Degree of superheating or subcooling 
        rho=D # Density in kg/m^3
        pressure=P_actual/10**5 # Pressure in barA
        Tempr=T_actual-273.15 #Temperatuire in °C



    elif input==3: # Temperature and gas mass fraction as input
        
        x= float(Input2)
        T_actual=273.15+float(Input1)

        P = CP.PropsSI('P', 'T', T_actual, 'Q', x, Refrigerant)
        rho = CP.PropsSI('D', 'T', T_actual, 'Q', x, Refrigerant)
        enthalpy = CP.PropsSI('H', 'T', T_actual, 'Q', x, Refrigerant)/1000  # Enthalpy in KJ/kg
        entropy = CP.PropsSI('S', 'T', T_actual, 'Q', x, Refrigerant)/1000  # Entropy in kJ/kg/K
        C_p_f = CP.PropsSI('C', 'T', T_actual, 'Q', 0, Refrigerant)       
        C_p_g= CP.PropsSI('C', 'T', T_actual, 'Q', 1, Refrigerant)
        C_p=  C_p_f+x*(C_p_g- C_p_f)      
        lambda_K = CP.PropsSI('L', 'T', T_actual, 'Q', x, Refrigerant)  # Thermal conductivity in W/m/K
        viscosity = CP.PropsSI('V', 'T', T_actual, 'Q', x, Refrigerant)*10**6  # Dynamic viscosity in Pa.s
        T_saturation = CP.PropsSI('T', 'P', P, 'Q', 0, Refrigerant)  # Saturation temperature of the liquid
        Superheat_Subcool=round(T_actual - T_saturation , 2)  # Degree of superheating or subcooling 
        pressure=P/10**5  # pressure in barA
        Tempr=T_actual-273.15 #Temperatuire in °C
    

    elif input==4: # Pressure and gas mass fraction as input
        x= float(Input2)
        P=float(Input1)*10**5


        T_actual = CP.PropsSI('T', 'P', P, 'Q', x, Refrigerant)

        rho = CP.PropsSI('D', 'P', P, 'Q', x, Refrigerant)
        enthalpy = CP.PropsSI('H', 'P', P, 'Q', x, Refrigerant)/1000  # Enthalpy in KJ/kg
        entropy = CP.PropsSI('S', 'P', P,'Q', x, Refrigerant)/1000  # Entropy in kJ/kg/K
        C_p_f = CP.PropsSI('C', 'P', P, 'Q', 0, Refrigerant)       
        C_p_g= CP.PropsSI('C', 'P', P, 'Q', 1, Refrigerant)
        C_p=  C_p_f+x*(C_p_g- C_p_f)
        lambda_K = CP.PropsSI('L', 'P', P,'Q', x, Refrigerant)  # Thermal conductivity in W/m/K
        viscosity = CP.PropsSI('V', 'P', P, 'Q', x, Refrigerant)*10**6  # Dynamic viscosity in Pa.s
        T_saturation = CP.PropsSI('T', 'P', P, 'Q', 0, Refrigerant)  # Saturation temperature of the liquid
        Superheat_Subcool=round(T_actual - T_saturation , 2)  # Degree of superheating or subcooling 
        pressure=P/10**5  # pressure in barA
        Tempr=T_actual-273.15 #Temperatuire in °C


    elif input==5: # Pressure and density as input
        P=float(Input1)*10**5  # Input pressure in Pa
        rho=float(Input2)      # Input density in kg/m3

        D=rho
        T_actual= CP.PropsSI('T', 'P', P, 'D', D, Refrigerant)
        print(T_actual)
        T_saturation = CP.PropsSI('T', 'P', P, 'Q', 0, Refrigerant) # Saturation temperature of the liquid
        Superheat_Subcool=round(T_actual - T_saturation , 2)  # Degree of superheating or subcooling 
        enthalpy = CP.PropsSI('H', 'P', P, 'D', D, Refrigerant)/1000   # Enthalpy in kJ/kg
        entropy = CP.PropsSI('S', 'P', P, 'D', D, Refrigerant)/1000   # Entropy in kJ/kg/K
        C_p = CP.PropsSI('C', 'P', P, 'D', D, Refrigerant)        # Specific heat in J/kg/K
        lambda_K = CP.PropsSI('L', 'P', P, 'D', D, Refrigerant)  # Thermal conductivity in W/m/K
        viscosity = CP.PropsSI('V', 'P', P, 'D', D, Refrigerant)*10**6  # Dynamic viscosity in µPa.s
        x= CP.PropsSI('Q','P', P, 'D', D, Refrigerant)  # Quality (0 = liquid, 1 = vapor)
        pressure=P/10**5 # Pressure in barA
        Tempr=T_actual-273.15 #Temperatuire in °C



    elif input==6: # Pressure and enthalpy as input
        P=float(Input1)*10**5
        H=float(Input2)

        T_actual= CP.PropsSI('T', 'P', P, 'H', H, Refrigerant)
        print(T_actual)
        T_saturation = CP.PropsSI('T', 'P', P, 'Q', 0, Refrigerant) # Saturation temperature of the liquid
        Superheat_Subcool=round(T_actual - T_saturation , 2)  # Degree of superheating or subcooling 
        entropy = CP.PropsSI('S', 'P', P, 'H', H, Refrigerant)/1000   # Entropy in kJ/kg/K
        C_p = CP.PropsSI('C', 'P', P, 'H', H,Refrigerant)        # Specific heat in J/kg/K
        lambda_K = CP.PropsSI('L', 'P', P, 'H', H,Refrigerant)  # Thermal conductivity in W/m/K
        viscosity = CP.PropsSI('V', 'P', P, 'H', H, Refrigerant)*10**6  # Dynamic viscosity in µPa.s
        x= CP.PropsSI('Q','P', P, 'H', H, Refrigerant)  # Quality (0 = liquid, 1 = vapor)
        pressure=P/10**5 # Pressure in barA
        rho = CP.PropsSI('D', 'P', P, 'H', H, Refrigerant)
        pressure=P/10**5 # Pressure in barA
        Tempr=T_actual-273.15 #Temperatuire in °C
        enthalpy=H/1000 # Enthalpy in KJ/kg
        

        
    elif input==7: # Pressure and subcooling as input
        P_actual=float(Input1)*10**5
        Superheat_Subcool=float(Input2)
        
  

        T_saturation = CP.PropsSI('T', 'P', P_actual, 'Q', 0, Refrigerant)  # Saturation temperature of the liquid
        T_actual=T_saturation + Superheat_Subcool

        D = CP.PropsSI('D', 'T', T_actual, 'P', P_actual, Refrigerant)

        enthalpy = CP.PropsSI('H', 'T', T_actual, 'P', P_actual, Refrigerant)/1000  # Enthalpy in kJ/kg
        entropy = CP.PropsSI('S', 'T', T_actual, 'P', P_actual, Refrigerant)/1000   # Entropy in kJ/kg/K
        C_p = CP.PropsSI('C', 'T', T_actual, 'P', P_actual, Refrigerant)        # Specific heat in J/kg/K
        lambda_K = CP.PropsSI('L', 'T', T_actual, 'P', P_actual, Refrigerant)  # Thermal conductivity in W/m/K
        viscosity= CP.PropsSI('V', 'T', T_actual, 'P', P_actual, Refrigerant)*10**6  # Dynamic viscosity in Pa.s
        x = CP.PropsSI('Q', 'T', T_actual, 'P', P_actual, Refrigerant)  # Quality (0 = liquid, 1 = vapor)
        T_saturation = CP.PropsSI('T', 'P', P_actual, 'Q', 0, Refrigerant)  # Saturation temperature of the liquid
        rho=D # Density in kg/m^3
        Tempr=T_actual-273.15 #Temperatuire in °C
        pressure=P_actual/10**5 # Pressure in barA



    elif input==8: # Pressure and entropy as input
        P=float(Input1)*10**5
        S=float(Input2)*1000

        T_actual= CP.PropsSI('T', 'P', P, 'S', S, Refrigerant)
        print(T_actual)
        T_saturation = CP.PropsSI('T', 'P', P, 'Q', 0, Refrigerant) # Saturation temperature of the liquid
        Superheat_Subcool=round(T_actual - T_saturation , 2)  # Degree of superheating or subcooling 
        enthalpy = CP.PropsSI('H', 'T', T_actual, 'S', S, Refrigerant)/1000  # Enthalpy in kJ/kg
        C_p = CP.PropsSI('C', 'P', P, 'S', S, Refrigerant)        # Specific heat in J/kg/K
        lambda_K = CP.PropsSI('L', 'P', P, 'S', S, Refrigerant)  # Thermal conductivity in W/m/K
        viscosity = CP.PropsSI('V', 'P', P, 'S', S, Refrigerant)*10**6  # Dynamic viscosity in µPa.s
        x= CP.PropsSI('Q','P', P, 'S', S, Refrigerant)  # Quality (0 = liquid, 1 = vapor)
        rho = CP.PropsSI('D', 'P', P, 'S', S, Refrigerant)
        Tempr=T_actual-273.15 #Temperatuire in °C
        pressure=P/10**5 # Pressure in barA
        entropy=S/1000




    if x!=-1:
        if x>=1 :
            Refrigerant_state='Vapour'
        elif x<=0 :
            Refrigerant_state='Liquid'
        elif x>0 and x<1:
            Refrigerant_state='Saturated liquid-vapour mixture'


    else:
        if Superheat_Subcool>0 :
            Refrigerant_state='Vapour'
            x=1
        else:
            Refrigerant_state='Liquid'
            x=0


    # print('Refrigerant state                 ', Refrigerant_state)
    # print('Pressure                          ',pressure,'bar' )
    # print('Density                           ',rho,'kg/m3')
    # print('Temperature                       ',Tempr,'°C')
    # print('Enthalpy                          ',enthalpy, 'kJ/kg' )
    # print('Entropy                           ',entropy,'kJ/K')
    # print('Cp                                ',C_p,'J/kg/K')
    # print('Thermal conductivity              ',lambda_K,'W/m K' )
    # print('Viscocity                         ',viscosity, 'µPa.S' )
    # print('Gas mass fraction                 ', x)
    # print('Superheating/Subcooling           ', Superheat_Subcool,'°C')



    return('R134a',Refrigerant_state,pressure,rho,Tempr,enthalpy,entropy,C_p,lambda_K,viscosity,x,Superheat_Subcool,)


#R134a(7,14.26,-10)