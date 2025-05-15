def EXV(m_dot_ref,Ps,Pd,rho_EXV_in):
    
    Cd=0.4
    Orifice_area=m_dot_ref/(Cd*(2*rho_EXV_in*(Pd-Ps)*100000)**0.5)*1000000
    Orifice_dia=((Orifice_area/3.124)**0.5)*2
    return(Orifice_area,Orifice_dia)