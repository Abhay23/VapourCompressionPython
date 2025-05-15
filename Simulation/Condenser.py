def Condenser( P_Cond_in,T_Cond_in,m_dot_ref,Condenser_dict,BoundaryConditions_dict):   #-200,100   
    import math 
    from sympy import symbols, Eq, solve, log
    import pandas as pd
    from coolprop import Refr

    Pair=101325         # Air pressure in Pa
    P_crit=4059280.0    # Critical pressure for R134a in Pa
    m_dot_air_total=BoundaryConditions_dict["M_air_cond"]/3600  # Air mass flow rate in kg/s
    CP_air=1000 # Specific heat capcity of air
    Tair=BoundaryConditions_dict["T_air_cond"]             # Temperature of air in °C


    # Function for Reynolds number of refrigerant
    def ReyN(G,D,mu):
        Re = G*D/mu
        Re=max(Re,1110)
        return Re

    #  Fanning friction factor in turbulent and laminar regions for & single-phase flow. Also calculates effective quantities for triangular &shape in order to scale calculations
    def Fri_F(Re_h,D_h,epsilon):
        f_f_l=13.33/Re_h # laminar friction factor, triangle
        D_le= 64/(4*f_f_l*Re_h)*D_h; #laminar equivelant diameter
        D_ic=1.5*D_h # inscribed-circumscribed diameter
        Re_le=Re_h*(D_le/D_h)
        Re_ic=Re_h*(D_ic/D_h)

        if (Re_le+Re_ic)/2 > 3000: 
            Re_eff = Re_ic
            D_eff = D_ic
        else:
            Re_eff = Re_le
            D_eff = D_le
        A = (2.457*math.log(1/((7/Re_h)**0.9+(0.27*epsilon/D_h))))**16
        B = (37530/Re_h)**16
        f_f = 2*(((8/Re_h)**12)+(1/(A+B)**(1.5)))**(1/12)
        return f_f,Re_eff,D_eff

    # Single phase Heat transfer coefficient
    def Refr_HTC_SP(f_f,T_ref,P_r,D_eff,L,L_m,A_ref_CS,Re_eff,k,ro,mu,cp):
        m = A_ref_CS*ro*L_m;
        Pr=mu*cp/k
        Nu = (4*f_f/8)*(Re_eff-1000)*Pr/(1+12.7*((4*f_f/8)**0.5)*((Pr**(2/3))-1))*(1+(D_eff/L)**(2/3))
        h=Nu*k/D_eff
        return h,m

    # Single phase Pressure drop 
    def Refr_DP_SP(f_f,L,D_h,G,T_ref,P_r,ro):
        v=1/ro # Specific volume
        dP= f_f*2*(G**2)*v*(L/D_h)
        return dP

    # Two phase Heat transfer coefficient
    def Refr_HTC_TP(P_ref,x1,D_h,L_m,A_port,G,T_air_i,twophase_code): 

        Refrigerant_a,Refrigerant_state_a,pressure_a,rho_a,Tempr_a,enthalpy_a,entropy_a, C_p_a, lambda_K_a,viscocity_fluid_a,x_a,Superheat_Subcool_a=Refr(BoundaryConditions_dict["Refrigerant"],4,P_ref,x1)
        Refrigerant_l,Refrigerant_state_l,pressure_l,rho_l,Tempr_l,enthalpy_l,entropy_l, C_p_l, lambda_K_l,viscocity_fluid_l,x_l,Superheat_Subcool_l=Refr(BoundaryConditions_dict["Refrigerant"],4,P_ref,0)
        Refrigerant_v,Refrigerant_state_v,pressure_v,rho_v,Tempr_v,enthalpy_v,entropy_v, C_p_v, lambda_K_v,viscocity_fluid_v,x_v,Superheat_Subcool_v=Refr(BoundaryConditions_dict["Refrigerant"],4,P_ref,1)

        P_sat=pressure_a*10**5
        mu_l=viscocity_fluid_l*10**-6
        mu_v=viscocity_fluid_v*10**-6
        cp_l=C_p_l
        cp_v=C_p_v
        k_l=lambda_K_l
        k_v=lambda_K_v
        ro_l=rho_l
        ro_v=rho_v
        Re_l_h=ReyN((G*(1-x1)),D_h,mu_l)
        Re_vo_h=ReyN(G,D_h,mu_v) # vapor only Reynolds number
        Re_lo_h= ReyN(G,D_h,mu_l) # liquid only Reynolds number

        Pr_l = mu_l*cp_l/k_l

        Xtt = (((1-x1)/x1)**0.875)*((mu_l/mu_v)**0.1)*((ro_v/ro_l)**0.5) # Lockhart-Martinelli Parameter
        Ft = 1 #math.sqrt((G**2*x1**3)/((1-x1)*ro_v**2*9.8*D_h)) # Froude rate
        alpha_newell = (1+1/Ft+Xtt)**(-0.321) #  Newell void fraction
        alpha_zivi = 1/(1+((1-x1)/x1)*(ro_v/ro_l)**(2/3)) # # Zivi void fraction 
        alpha_butterworth = 1/(1+0.28*(((1-x1)/x1)**0.64)*((ro_v/ro_l)**(0.36))*((mu_l/mu_v)**(0.07))) # Butterworth void fraction
        alpha = alpha_newell
        m = A_port*L_m*(ro_v*alpha+ro_l*(1-alpha)) # module refrigerant mass

        Nu_Shah = 0.023*(Re_lo_h**0.8)*(Pr_l**0.4)*(((1-x1)**(0.8))+(3.8*(x1**0.76)*(1-x1)**0.04)/((P_sat/P_crit)**0.38)) # Shah correlation for heat transfer
        h=Nu_Shah*k_l/D_h
        return h,Re_l_h,m

    # Two phase Pressure drop 
    def Refr_DP_TP(L,D_h,G,T_ref,P_ref,x1,x2, epsilon):
        if x2 < 0.001: 
            dP=0     
        else:
            Refrigerant_a,Refrigerant_state_a,pressure_a,rho_a,Tempr_a,enthalpy_a,entropy_a, C_p_a, lambda_K_a,viscocity_fluid_a,x_a,Superheat_Subcool_a=Refr(BoundaryConditions_dict["Refrigerant"],4,P_ref,x1)
            Refrigerant_l,Refrigerant_state_l,pressure_l,rho_l,Tempr_l,enthalpy_l,entropy_l, C_p_l, lambda_K_l,viscocity_fluid_l,x_l,Superheat_Subcool_l=Refr(BoundaryConditions_dict["Refrigerant"],4,P_ref,0)
            Refrigerant_v,Refrigerant_state_v,pressure_v,rho_v,Tempr_v,enthalpy_v,entropy_v, C_p_v, lambda_K_v,viscocity_fluid_v,x_v,Superheat_Subcool_v=Refr(BoundaryConditions_dict["Refrigerant"],4,P_ref,1)
            
            mu_l=viscocity_fluid_l*10**-6
            mu_v=viscocity_fluid_v*10**-6
            ro_l=rho_l
            ro_v=rho_v

            Re_lo= ReyN(G,D_h,mu_l)
            x_ave = (x1+x2)/2 # average quality of module
            alpha1 = 1/(1+((1-x1)/x1)*(ro_v/ro_l)**(2/3)) # Zivi void fraction, module inlet
            alpha2 = 1/(1+((1-x2)/x2)*(ro_v/ro_l)**(2/3)) # Zivi void  fraction, module outlet
            Xtt = (((1-x_ave)/x_ave)**(0.875))*((ro_v/ro_l)**(0.5))*((mu_l/mu_v)**(0.125)) #Lockhart-Martinelli parameter
            Gamma = ((ro_l/ro_v)**(0.5))*((mu_v/mu_l)**(0.125)) # Physical property index
            #f_lo = (1/12.96)/((math.log10((6.9/Re_lo)+(epsilon/D_h/3.7)**1.11))**2) # liquid only Fanning friction factor from Haaland
            f_lo = 0.079/(Re_lo**0.25) # liquid only Fanning friction factor from Blasius correlation
            DP_lo = 2*f_lo*(G**2)*L/ro_l/D_h # liquid only frictional pressure drop
            B = x2+0.363636*x2**(11/4)*Gamma**2+0.34632727*x2**(11/4)*Gamma**3*Xtt**(2063/5000)-0.363636*x2**(11/4)-0.34632727*x2**(11/4)*Gamma*Xtt**(2063/5000)
            A = x1+0.363636*x1**(11/4)*Gamma**2+0.34632727*x1**(11/4)*Gamma**3*Xtt**(2063/5000)-0.363636*x1**(11/4)-0.34632727*x1**(11/4)*Gamma*Xtt**(2063/5000)

            # overall pressure drop due to friction
            if x2==x1: 
                DP_f = DP_lo*(1+(Gamma**2-1)*(x1**1.75)*(1+0.9524*Gamma*Xtt**0.4126))  # no integral needed, adiabatic
            else: 
                DP_f = DP_lo/(x2-x1)*(B-A)
                print('dP_F',DP_f )
            # pressure drop (will be neg. for condensation) due to acceleration
            if alpha2 == 0:
                DP_ac = (G**2)*(-(((x1**2)/(ro_v*alpha1))+(((1-x1)**2)/(ro_l*(1-alpha1)))))
            elif alpha2 ==1:
                DP_ac = (G**2)*((((x2**2)/(ro_v*alpha2)))-(((x1**2)/(ro_v*alpha1))+(((1-x1)**2)/(ro_l*(1-alpha1)))))
            else:
                DP_ac = (G**2)*((((x2**2)/(ro_v*alpha2))+(((1-x2)**2)/(ro_l*(1-alpha2))))-(((x1**2)/(ro_v*alpha1))+(((1-x1)**2)/(ro_l*(1-alpha1)))))
                print('dP_ac',DP_ac )
            dP= DP_f + DP_ac
        return dP

    # Select heat transfer and pressure drop equations based upon phase of refrigerant
    def h_ref_select(T_ref,P_ref,x1,x2,D_h,G,epsilon,L,L_cnd,A_port,T_air_i,twophase_code):
        # Single phase
        if x1 == 0 or x1==1:
            P_ref=P_ref
            k=lambda_K # Thermal condiuctivity in W/m/K
            ro= rho # Density in kg/m3 at 20 bar, 80°C
            mu= viscocity_fluid*10**-6 # Dynamic viscocity
            cp= C_p # Heat capacity in J/kg/k

            Re_h=ReyN(G,D_h,mu) #Reynolds number
            f_f,Re_eff,D_eff=Fri_F(Re_h,D_h,epsilon) #friction factor

            h_c_ref,m =Refr_HTC_SP(f_f,T_ref,P_ref,D_eff,L_cnd,L,A_port,Re_eff,k,ro,mu,cp) # single-phase heat transfer coefficient and module refrigerant mass
            dP=Refr_DP_SP(f_f,L,D_h,G,T_ref,P_ref,ro)
            P_next = P_ref - dP/(10**5)


        #two phase
        else:
            h_c_ref,Re_eff,m=Refr_HTC_TP(P_ref,x1,D_h,L,A_port,G,T_air_i,twophase_code) # two-phase heat transfer coefficient and module refrigerant mass
            dP=Refr_DP_TP(L,D_h,G,T_ref,P_ref,x1,x2,epsilon) # module two-phase pressure drop
            P_next = P_ref - dP/(10**5)




        return h_c_ref,P_next,m

    # air side heat transfer coefficient [W/m**2-K] from the Chang-Wang 1996 correlation for Multi-louver fin geometry
    def h_air(Tair,Pair,vel,AR,ThetaLo,Lp,Fp,F1,Td,L1,Tp,Finth,K_fin):


        P_vs_air=133.322*10**(8.07131-1730.63/(233.426+BoundaryConditions_dict["T_air_cond"])) # Saturation vapour pressure in Pa
        P_v_air=BoundaryConditions_dict["RH_air_cond"]/100*P_vs_air # Partial vapour pressure in Pa
        rho_w_air=0.622*P_v_air/(BoundaryConditions_dict["P_air_cond"]*10**5-P_v_air) # Absolute humidity of moist air
        Molar_mass_air=(1-rho_w_air)*28.97+18.02*rho_w_air # Molar mass of moist air
        rho_air=BoundaryConditions_dict["P_air_cond"]*10**5/(Molar_mass_air*8.314*(BoundaryConditions_dict["T_air_cond"]+273.15))   #density of moist air
        mu_air=1.716*10**-5*((BoundaryConditions_dict["T_air_cond"]+273.15)/273.15)**1.5*(273.15+111)/(BoundaryConditions_dict["T_air_cond"]+273.15+111)*(1+0.02*rho_w_air)
        k_air=0.0242*((BoundaryConditions_dict["T_air_cond"]+273.15)/273.15)**1.5*(273.15+194)/(BoundaryConditions_dict["T_air_cond"]+273.15+194)*(1+0.01*rho_w_air)
        cp_air=1005+0.1*(BoundaryConditions_dict["T_air_cond"]+273.15)-0.0003*(BoundaryConditions_dict["T_air_cond"]+273.15)**2+rho_w_air*1860

        Re_air=vel*rho_air*Lp/mu_air; # Reynold number heat of air
        ff=(ThetaLo/90)**0.27*(Fp/Lp)**(-0.14)*(F1/Lp)**(-0.29)*(Td/Lp)**(-0.23)*(L1/Lp)**0.68*(Tp/Lp)**(-0.28)*(Finth/Lp)**(-0.05); # friction factor
        n=0.49;
        j=Re_air**(-n)*ff; # Colburn factor
        Pr_air=mu_air*cp_air/k_air;
        St=j*Pr_air**(-2/3); # Stanton Number
        h_air=St*rho_air*vel*cp_air;
        ML=(2*h_air/(K_fin*Finth))**0.5*(F1/2-Finth);
        etaf=math.tanh(ML)/(ML) # Fin efficiency
        etaa=1-AR*(1-etaf)
        return h_air,etaf,etaa,Re_air

    # Calculation of condneser geometry
    # Major dimensions
    H_cond=Condenser_dict["H_cond"]/1000 # condenser height [m]
    W_cond=Condenser_dict["W_cond"]/1000 # condenser width [m] (between headers)
    Depth_cond=Condenser_dict["Depth_cond"]/1000  # condenser slab depth [m]
    # Tubes geometry
    N_passes = Condenser_dict["N_passes"] #number of passes
    N_ports = Condenser_dict["N_ports"] # number of ports in each tube
    N_tubes=[]
    for zz in range(0,N_passes):
        N_tubes.append(int(Condenser_dict["Tubes_split"].split("-") [zz])) # number of tubes in each pass
    D_char = Condenser_dict["D_char"]/1000 # height of port triangles [m]
    Thickness_tube=Condenser_dict["Thickness_tube"]/1000 # tube minor diameter [m]
    N_tube=Condenser_dict["N_tube"] # total number of tubes
    Depth_tube=Depth_cond # tube major diameter[m]
    # Fin geometry
    P_fin=Condenser_dict["P_fin"]/1000# fin pitch [m]
    N_fpc=round(W_cond/P_fin) # number of fins in condenser width
    H_fin=(H_cond-N_tube*Thickness_tube)/(N_tube+1)# fin height, dismath.tance between the tubes [m]
    Depth_fin=Depth_cond # depth of fin, core depth [m]
    Thickness_fin=Condenser_dict["Thickness_fin"]/1000 # thickness of fin [m]
    K_fin=174 # conductivity of fin material [W/m-K]
    P_tube= H_fin+Thickness_tube # tube pitch [m]
    #Fin louver geometry
    P_louver=Condenser_dict["P_louver"]/1000  # louver pitch [m]
    L_louver=Condenser_dict["L_louver"]/1000  # louver length [m]
    Theta_louver=Condenser_dict["Theta_louver"]  # louver angle [deg]
    # Header geometry
    H_header = 0.254 # height of header interior [m] 
    L_outlet_from_bottom = 1.25*(.0254) # distance of outlet port of second pass from bottom of condenser [m]
    D_header = 19/1000 # inner diameter of header sections [m]
    epsilon_port = 0.000005 # absolute roughness of port tube length
        

    # Calulation of macro dimmensions
    A_fin_CS=H_fin*Thickness_fin*N_fpc*(N_tube+1)  #crosssectional area of fins as exposed to air flow                             fins frontal area
    A_tubes_CS=W_cond*Thickness_tube*N_tube  # crosssectional area of tubes as exposed to air flow                                 tubes frontal atrea
    A_cond_CS=H_cond*W_cond  # total conenser face area, or frontal face area                                                      frontal area
    A_air_ff=A_cond_CS -A_tubes_CS-A_fin_CS    # condenser air side free flow area                                                 not there
    Volume_core=A_cond_CS*Depth_cond # condenser core volume                                                                       not there
    A_air_tubes=W_cond*Depth_tube*2*(N_tube+1) # heat transfer tube area exposed to flow                                           tubes longitudinal area, N_tube+1                 
    A_air_fin=Depth_fin*H_fin*N_fpc*2*(N_tube+1)  # heat transfer fin area exposed to flow                                         fins longitudinal area  
    A_air_total=A_air_tubes+A_air_fin # condenser total air side area                                                              external convective area
    D_port_h = 4*(D_char**2)/math.tan(60*3.124/180)/(3*D_char/math.sin(60*3.124/180))  # hydraulic diameter of refrigerant free flow area               hydraulic diameter 4Ac/P
    A_ref = 3*D_char/math.sin(60*3.124/180)*N_ports*W_cond*N_tube   # total refrigerant surface area of all tubes                             microchannels convective area
    A_ref_CS_total= (D_char**2)/math.tan(60*3.124/180)*N_ports*N_tube     # total refrigerant cross sectional area of all tubes     microchannels cross-section area
    AR=A_air_fin/A_air_total
    RR=2/3 # Aweb over Aref
    eta_fin_air =0.2 # math.tanh(H_fin/2*math.sqrt(2*h_c_air/K_fin/Thickness_fin))/(H_fin/2*math.sqrt(2*h_c_air/K_fin/Thickness_fin))   # fin efficiency
    eta_air_o = 1-AR*(1-eta_fin_air) #air side surface efficiency, this is multiplied by heat transfer &coefficient
    Thickness_web = 0.251/1000    # tube side web thickness (wall thickness between ports)
    L_web = 1.636/1000 #length (or height) of the refrigerant web
    N_modules = 30 # number of modules--finite difference mesh control volumes--in each pass


    # length--refrigerant flow direction of each module
    L_modules = W_cond/N_modules
    Volume_cnd = 2*(0.25*3.124*D_header**2)*H_cond+A_ref_CS_total*W_cond # total internal volume of condenser, including headers     

    # calculation for quantities that do not vary within each pass
    A_air=[]
    A_ref_port=[]
    m_dot_air=[]
    A_ref_CS=[]
    G=[]

    for i in range(0,N_passes):
        A_air.append(A_air_total*(N_tubes[i]/N_tube)/N_modules) # module air surface area
        A_ref_port.append(A_ref*(N_tubes[i]/N_tube)/N_modules) # module refrigerant surface area
        m_dot_air.append(m_dot_air_total*(N_tubes[i]/N_tube)/N_modules) # module air mass flow rate
        A_ref_CS.append(A_ref_CS_total*(N_tubes[i]/N_tube)) # refrigerant free flow area
        G.append(m_dot_ref/A_ref_CS[i]) # mass flux

    # air specific volume m^3/kg]
    v_air= 1/1.164 #VOLUME(Air,T=T_air_i_louver,P=P_n_i+1.5*DP_Cond)
    Vel_air=m_dot_air_total*v_air/A_air_ff # air velocity over louvers, [m/sec]

    # air side heat transfer coefficient from Chang-Wang correlation, [W/m^2-C]}
    h_c_air,eta_fin,eta_sur,Re_air= h_air(Tair,Pair,Vel_air,AR,Theta_louver,P_louver,P_fin,H_fin,Depth_tube,L_louver,P_tube,Thickness_fin,K_fin)


    # State of refrigerant at condenser inlet/1st Module
    T_ref=[T_Cond_in] # Refrigerant inlet temperature
    P_ref=[P_Cond_in] # Refrigerant inlet pressure
    Refrigerant,Refrigerant_state,pressure,rho,Tempr,enthalpy,entropy, C_p, lambda_K,viscocity_fluid,x,Superheat_Subcool=Refr(BoundaryConditions_dict["Refrigerant"],2,T_ref[0],P_ref[0])
    x_ref=[x] # Refrigerant inlet gas mass fraction
    h_ref=[enthalpy*1000]  # Refrigerant inlet ebnthalpy
    Subcool_ref=[Superheat_Subcool] # Refrigerant inlet superheat
    CP_ref=[C_p]

    # Refrigerant properties that need to be calculated for each module
    h_c_ref=[]   # Heat transfer coefficient of each module
    m_ref=[]     # Mass of refrigernat in each module
    m=[]        # Air mass flow of of each module
    UA=[]       # UA of of each module
    Q_dot=[]    # Heat transfer for each module
    T_air_o=[]  # Air oputlet temperature for each module


    T_air_i_louver=Tair
    twophase_code=4
    

    for k in range(0,N_passes):

        for j in range(N_modules*k,N_modules*(k+1)):

            
            if x_ref[j]>0.0001 and x_ref[j]<0.9999:
                x2=x_ref[j]-0.0001
                for i in range(1,20):
                    h_c_ref_calc,P_ref_calc,m_ref_calc= h_ref_select(T_ref[j],P_ref[j],x_ref[j],x2,D_port_h, G[k],epsilon_port,L_modules,W_cond,A_ref_CS[k],T_air_i_louver,twophase_code)
                    UA_calc = 1/((1/(h_c_air*eta_air_o))+(1/(h_c_ref_calc*(1-RR*(1-math.tanh(L_web/2*math.sqrt(2*h_c_ref_calc/K_fin/Thickness_web)) /(L_web/2*math.sqrt(2*h_c_ref_calc/K_fin/Thickness_web))))*A_ref_port[k]/A_air[k])))*A_air[k] # guessed overall hTC' 
                    NTU=UA_calc/(m_dot_air[k]*CP_air)
                    eff=1-math.exp(-1*NTU)
                    Q_dot_calc=eff*m_dot_air[k]*CP_air*(T_ref[j]-T_air_i_louver)
                    h_ref_calc=h_ref[j]-Q_dot_calc/m_dot_ref

                    Refrigerant,Refrigerant_state,pressure,rho,Tempr,enthalpy,entropy, C_p, lambda_K,viscocity_fluid,x,Superheat_Subcool=Refr(BoundaryConditions_dict["Refrigerant"],6,P_ref_calc,h_ref_calc)
                    if abs(x2 - x) < 0.001:
                        break 
                    else:
                        x2=(x+x2)/2
                    
                
            else:  
                h_c_ref_calc,P_ref_calc,m_ref_calc= h_ref_select(T_ref[j],P_ref[j],x_ref[j],x_ref[j],D_port_h, G[k],epsilon_port,L_modules,W_cond,A_ref_CS[k],T_air_i_louver,twophase_code)
                UA_calc = 1/((1/(h_c_air*eta_air_o))+(1/(h_c_ref_calc*(1-RR*(1-math.tanh(L_web/2*math.sqrt(2*h_c_ref_calc/K_fin/Thickness_web)) /(L_web/2*math.sqrt(2*h_c_ref_calc/K_fin/Thickness_web))))*A_ref_port[k]/A_air[k])))*A_air[k] # guessed overall hTC' 
                NTU=UA_calc/min(CP_air*m_dot_air[k],CP_ref[j]*m_dot_ref)

                if (CP_air*m_dot_air[k])>(CP_ref[j]*m_dot_ref):
                    Cr=(CP_ref[j]*m_dot_ref)/(CP_air*m_dot_air[k])
                    eff=1/Cr*(1-math.exp(-1*Cr*(1-math.exp(-1*NTU))))
                    
                else:
                    Cr=(CP_air*m_dot_air[k])/(CP_ref[j]*m_dot_ref)
                    eff=1-math.exp(-1/Cr*(1-math.exp(-1*Cr*NTU)))

                Q_dot_calc=eff*min(CP_air*m_dot_air[k],CP_ref[j]*m_dot_ref)*(T_ref[j]-T_air_i_louver)
                h_ref_calc=h_ref[j]-Q_dot_calc/m_dot_ref
            
                Refrigerant,Refrigerant_state,pressure,rho,Tempr,enthalpy,entropy, C_p, lambda_K,viscocity_fluid,x,Superheat_Subcool=Refr(BoundaryConditions_dict["Refrigerant"],6,P_ref_calc,h_ref_calc)
            

            T_air_o_calc=T_air_i_louver+Q_dot_calc/(m_dot_air[k]*CP_air)

            Subcool_ref.append(round(Superheat_Subcool,2))
            T_ref.append(round(Tempr,3))
            x_ref.append(round(x,3))
            h_ref.append(round(h_ref_calc,3))
            P_ref.append(round(P_ref_calc,3))
            CP_ref.append(round(C_p,3))
            h_c_ref.append(round(h_c_ref_calc,3))
            Q_dot.append(round(Q_dot_calc,3))
            UA.append(round(UA_calc,3))
            m_ref.append(round(m_ref_calc,3))
            T_air_o.append(round(T_air_o_calc,3))

    # Converting to float
    T_air_o = [float(x) for x in T_air_o]
    h_ref = [float(x) for x in h_ref]
    P_ref = [float(x) for x in P_ref]
    h_c_ref = [float(x) for x in h_c_ref]
    UA = [float(x) for x in UA]
    G = [float(x) for x in G]
    Subcool_ref = [float(x) for x in Subcool_ref]        
                
    Q_dot_total=sum(Q_dot)
    T_air_out_avg=sum(T_air_o)/59

    print('T_air_out_avg',T_air_out_avg)
    print('Q_dot_total',Q_dot_total)
    print('Q_dot',Q_dot)
    print('x_ref',x_ref)
    print('T_ref',T_ref)
    print('h_ref',h_ref)
    print('P_ref',P_ref)
    print('Subcool',Subcool_ref)

    print('h_c_ref',h_c_ref)
    print('h_c_air',h_c_air)
    print('T_air_o',T_air_o)
    print('UA',UA)

    print('A_air',A_air)
    print('m_dot_air',m_dot_air)
    print('A_ref_CS',A_ref_CS)
    print('G',G)


    return(P_ref[j],h_ref[j],Q_dot_total,x_ref[j],Subcool_ref[j],T_air_out_avg)

