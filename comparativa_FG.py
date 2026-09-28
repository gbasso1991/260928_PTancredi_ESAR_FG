#%% Librerias y paquetes
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from scipy.interpolate import interp1d
from clase_resultados import ResultadosESAR
#%% Lector de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))

                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor

                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype= {'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% funcion extraer SAR, tau y Hc de resultados
def extraer_SAR_tau(resultados):
    SAR = []
    tau = []
    Hc = []
    for res in resultados:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(res)
        SAR.append(meta['SAR_W/g'])
        tau.append(meta['tau_ns'])
        Hc.append(meta['Hc_kA/m'])
    return SAR, tau, Hc
#%% funcion banda temperatura
def banda_temperatura(t, T, N=500, kind='linear'):
    """
    Interpola varias curvas T(t) sobre una grilla temporal común y
    calcula estadísticas punto a punto.

    Parameters
    ----------
    t : list of np.ndarray
        Lista de vectores de tiempo.
    T : list of np.ndarray
        Lista de vectores de temperatura.
    N : int, optional
        Número de puntos de la grilla común.
    kind : str, optional
        Tipo de interpolación (interp1d).

    Returns
    -------
    tt : list of np.ndarray
        Lista original de tiempos.
    TT : list of np.ndarray
        Lista original de temperaturas.
    t_common : np.ndarray
        Grilla temporal común.
    Tmin : np.ndarray
        Temperatura mínima en cada instante.
    Tmax : np.ndarray
        Temperatura máxima en cada instante.
    Tmean : np.ndarray
        Temperatura promedio en cada instante.
    """

    # intervalo temporal común
    tmin = max(tt.min() for tt in t)
    tmax = min(tt.max() for tt in t)

    t_common = np.linspace(tmin, tmax, N)

    # interpolación
    Ti = []
    for tt, TT in zip(t, T):
        f = interp1d(tt, TT, kind=kind)
        Ti.append(f(t_common))

    Ti = np.asarray(Ti)

    # estadísticas
    Tmin  = np.min(Ti, axis=0)
    Tmax  = np.max(Ti, axis=0)
    Tmean = np.mean(Ti, axis=0)

    return t, T, t_common, Tmin, Tmax, Tmean
#%% FG Co
nombre_Co='Co'
ciclos_Co = glob("Co/**/*ciclo_promedio_H_M.txt", recursive=True)
resultados_Co = glob("Co/**/*resultados.txt", recursive=True)
ciclos_Co.sort()
resultados_Co.sort()
conc_Co =  12.0 #g/L 

print('Importando ciclos de', nombre_Co,'\n')
for p in ciclos_Co:
    print('  ',p)
print('\n')
for res in resultados_Co:
    print('  ',res)
print('-'*50)

SAR_Co, tau_Co, Hc_Co = extraer_SAR_tau(resultados_Co)
res_Co=[]
#%% ploteo ciclos
fig01, axs =plt.subplots(1,2,figsize=(12,6),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_ylabel('M (A/m)')

axs[0].set_title('170 kHz',loc='left')
axs[1].set_title('265 kHz',loc='left')

for i,e in enumerate(ciclos_Co):
    if '170kHz' in e:
        _,_,_, H_Co,M_Co,_ = lector_ciclos(ciclos_Co[i])
        print('1',os.path.basename(e))
        axs[0].plot(H_Co/1000,M_Co,'-',label=f'{SAR_Co[i]:.3uS}')

    elif '265kHz' in e:
        _,_,_, H_Co,M_Co,_ = lector_ciclos(ciclos_Co[i])
        print('2',os.path.basename(e))
        axs[1].plot(H_Co/1000,M_Co,'-',label=f'{SAR_Co[i]:.3uS}')

for a in axs:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Ciclos promedio {nombre_Co} \n170 & 265 kHz - 23 & 46 kA/m\nC = {conc_Co:.1f} g/L')


################################################################################################################################
#%% 2- Fe 
nombre_Fe='Fe'
ciclos_Fe = glob("Fe/**/*ciclo_promedio_H_M.txt", recursive=True)
resultados_Fe = glob("Fe/**/*resultados.txt", recursive=True)
ciclos_Fe.sort()
resultados_Fe.sort()
conc_Fe =  9.0 #g/L 

print('Importando ciclos de', nombre_Fe,'\n')
for p in ciclos_Fe:
    print('  ',p)
print('\n')
for res in resultados_Fe:
    print('  ',res)
print('-'*50)

SAR_Fe, tau_Fe, Hc_Fe = extraer_SAR_tau(resultados_Fe)
res_Fe=[]
#%% ploteo ciclos
fig02, axs =plt.subplots(1,2,figsize=(12,6),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_ylabel('M (A/m)')
axs[0].set_ylabel('M (A/m)')

axs[0].set_title('170 kHz',loc='left')
axs[1].set_title('265 kHz',loc='left')

for i,e in enumerate(ciclos_Fe):
    if '170kHz' in e:
        _,_,_, H_Fe,M_Fe,_ = lector_ciclos(ciclos_Fe[i])
        print('1',os.path.basename(e))
        axs[0].plot(H_Fe/1000,M_Fe,'-',label=f'{SAR_Fe[i]:.3uS}')

    elif '265kHz' in e:
        _,_,_, H_Fe,M_Fe,_ = lector_ciclos(ciclos_Fe[i])
        print('2',os.path.basename(e))
        axs[1].plot(H_Fe/1000,M_Fe,'-',label=f'{SAR_Fe[i]:.3uS}')

for a in axs:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Ciclos promedio {nombre_Fe} \n170 & 265 kHz - 23 & 46 kA/m\nC = {conc_Fe:.1f} g/L')

################################################################################################################################
#%% 3- Zn 
nombre_Zn='Zn'
ciclos_Zn = glob("Zn/**/*ciclo_promedio_H_M.txt", recursive=True)
resultados_Zn = glob("Zn/**/*resultados.txt", recursive=True)
ciclos_Zn.sort()
resultados_Zn.sort()
conc_Zn =  12.0 #g/L 

print('Importando ciclos de', nombre_Zn,'\n')
for p in ciclos_Zn:
    print('  ',p)
print('\n')
for res in resultados_Zn:
    print('  ',res)
print('-'*50)

SAR_Zn, tau_Zn, Hc_Zn = extraer_SAR_tau(resultados_Zn)
res_Zn=[]
#%% ploteo ciclos
fig03, axs =plt.subplots(1,2,figsize=(12,6),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_ylabel('M (A/m)')
axs[0].set_ylabel('M (A/m)')

axs[0].set_title('170 kHz',loc='left')
axs[1].set_title('265 kHz',loc='left')

for i,e in enumerate(ciclos_Zn):
    if '170kHz' in e:
        _,_,_, H_Zn,M_Zn,_ = lector_ciclos(ciclos_Zn[i])
        print('1',os.path.basename(e))
        axs[0].plot(H_Zn/1000,M_Zn,'-',label=f'{SAR_Zn[i]:.3uS}')

    elif '265kHz' in e:
        _,_,_, H_Zn,M_Zn,_ = lector_ciclos(ciclos_Zn[i])
        print('2',os.path.basename(e))
        axs[1].plot(H_Zn/1000,M_Zn,'-',label=f'{SAR_Zn[i]:.3uS}')

for a in axs:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Ciclos promedio {nombre_Zn} \n170 & 265 kHz - 23 & 46 kA/m\nC = {conc_Zn:.1f} g/L')

#%% Salvo figuras

fig01.savefig('01_ciclos_265-170_46-29_Co.png',dpi=300)
fig02.savefig('01_ciclos_265-170_46-29_Fe.png',dpi=300)
fig03.savefig('01_ciclos_265-170_46-29_Zn.png',dpi=300)


#%% Normalizo ciclos por concentracion y ploteo comparativo

fig2, axs =plt.subplots(1,3,figsize=(16,5),constrained_layout=True,sharey=False,sharex=False)
axs[0].set_ylabel('M (A/m)')
axs[0].set_title('38 kA/m',loc='left')
axs[1].set_title('47 kA/m',loc='left')
axs[2].set_title('57 kA/m',loc='left')

for i,e in enumerate(ciclos_M15):
    if '100dA' in e:
        _,_,_, H_M15,M_M15,_ = lector_ciclos(ciclos_M15[i])
        print('1.1',os.path.split(e)[-1])
        axs[0].plot(H_M15/1000,M_M15/conc_M15,'-',c='C0',label=f'M15\n{conc_M15} g/L' if i==0 else "")
    elif '125dA' in e:
        _,_,_, H_M15,M_M15,_ = lector_ciclos(ciclos_M15[i])
        print('2.1',os.path.split(e)[-1])
        axs[1].plot(H_M15/1000,M_M15/conc_M15,'-',c='C0',label=f'M15\n{conc_M15} g/L' if i==4 else "")
    elif '150dA' in e:
        _,_,_, H_M15,M_M15,_ = lector_ciclos(ciclos_M15[i])
        print('3.1',os.path.split(e)[-1])
        axs[2].plot(H_M15/1000,M_M15/conc_M15,'-',c='C0',label=f'M15\n{conc_M15} g/L' if i==7 else "")
        
for i,e in enumerate(ciclos_Fe):
    if '100dA' in e:
        _,_,_, H_Fe,M_Fe,_ = lector_ciclos(ciclos_Fe[i])
        print('1.2',os.path.split(e)[-1])
        axs[0].plot(H_Fe/1000,M_Fe/conc_Fe,'-',c='C1',label=f'Fe\n{conc_Fe} g/L' if i==0 else "")
    elif '125dA' in e:
        _,_,_, H_Fe,M_Fe,_ = lector_ciclos(ciclos_Fe[i])
        print('2.2',os.path.split(e)[-1])
        axs[1].plot(H_Fe/1000,M_Fe/conc_Fe,'-',c='C1',label=f'Fe\n{conc_Fe} g/L' if i==4 else "")
    elif '150dA' in e:
        _,_,_, H_Fe,M_Fe,_ = lector_ciclos(ciclos_Fe[i])
        print('3.2',os.path.split(e)[-1])
        axs[2].plot(H_Fe/1000,M_Fe/conc_Fe,'-',c='C1',label=f'Fe\n{conc_Fe} g/L' if i==7 else "")

axs[0].set_ylabel('M/[NPM] (Am²/kg)')
for a in axs:
    a.set_xlabel('H (kA/m)')
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,ncol=2)
plt.suptitle(f'Ciclos promedio nomalizados por concentracion\n300 kHz & [38, 47, 57] kA/m\n')

#%% ploteo comparativo de errorbars de ESAR
categorias = ['M15', 'Fe']
x = np.arange(len(categorias))

fig3, (ax,ax2,ax3) = plt.subplots(1,3,figsize=(12,4),constrained_layout=True,sharey=True)

sep = 0.25

for i,s in enumerate(SAR_M15[:3]):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(SAR_Fe[:3]):
    ax.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for i,s in enumerate(SAR_M15[3:7]):
    ax2.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(SAR_Fe[3:7]):
    ax2.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for i,s in enumerate(SAR_M15[7:]):
    ax3.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(SAR_Fe[7:]):
    ax3.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')
for a in [ax,ax2,ax3]:
    a.grid(axis='y', alpha=0.3)
    a.set_xticks(x)
    a.set_xticklabels(categorias)
    

ax.set_ylabel('ESAR (W/g)')
plt.suptitle(f'ESAR\n300 kHz & [38, 47, 57] kA/m\n')

plt.show()
#%% ploteo comparativo de tau

categorias = ['260630\nNF@cit', '260630\nNF@Fe']
x = np.arange(len(categorias))

fig4, (ax,ax2,ax3) = plt.subplots(1,3,figsize=(12,4),constrained_layout=True,sharey=True)

sep = 0.25

for i,s in enumerate(tau_M15[:3]):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(tau_Fe[:3]):
    ax.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for i,s in enumerate(tau_M15[3:7]):
    ax2.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(tau_Fe[3:7]):
    ax2.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for i,s in enumerate(tau_M15[7:]):
    ax3.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(tau_Fe[7:]):
    ax3.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')
for a in [ax,ax2,ax3]:
    a.grid(axis='y', alpha=0.3)
    a.set_xticks(x)
    a.set_xticklabels(categorias)
ax.set_ylabel('tau (ns)')
plt.suptitle(f'tau\n300 kHz & [38, 47, 57] kA/m\n')
plt.show()

#%% Idem Hc
fig5, (ax,ax2,ax3) = plt.subplots(1,3,figsize=(12,4),constrained_layout=True,sharey=True)

sep = 0.25

for i,s in enumerate(Hc_M15[:3]):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(Hc_Fe[:3]):
    ax.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for i,s in enumerate(Hc_M15[3:7]):
    ax2.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(Hc_Fe[3:7]):
    ax2.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for i,s in enumerate(Hc_M15[7:]):
    ax3.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(Hc_Fe[7:]):
    ax3.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')
for a in [ax,ax2,ax3]:
    a.grid(axis='y', alpha=0.3)
    a.set_xticks(x)
    a.set_xticklabels(categorias)
ax.set_ylabel('Hc (kA/m)')
plt.suptitle(f'Hc\n300 kHz & [38, 47, 57] kA/m\n')
plt.show()
#%% Salvo todas las figuras
fig00.savefig('00_ciclos_promedio_M15.png',dpi=300)
fig10.savefig('00_ciclos_promedio_Fe.png',dpi=300)
fig000.savefig('01_ciclos_promedio_all_M15.png',dpi=300)
fig100.savefig('01_ciclos_promedio_all_Fe.png',dpi=300)
fig01.savefig('02_templogs_M15.png',dpi=300)
fig11.savefig('03_templogs_Fe.png',dpi=300)
fig2.savefig('04_ciclos_promedio_comparativa.png',dpi=300)
fig3.savefig('05_ESAR_comparativa.png',dpi=300)
fig4.savefig('06_tau_comparativa.png',dpi=300)
fig5.savefig('07_Hc_comparativa.png',dpi=300)

# %%
#%% Printeo resultados
print(f'Muestra = {nombre_M15}')
print(f'Concentracion = {conc_M15:.1f} g/L')
print(f'ESAR = {SAR_M15} W/g')
print(f'tau = {tau_M15} ns')
print(f'Hc = {Hc_M15} kA/m') 
print(f'WR = {rates_M15} °C/s')

#%%
def promedio_por_grupos(lista, tamanos):
    """
    Calcula el promedio de grupos de distinto tamaño.
    
    tamanos: lista con la cantidad de elementos de cada grupo.
    """
    promedios = []
    inicio = 0

    for n in tamanos:
        grupo = lista[inicio:inicio+n]
        promedios.append(sum(grupo) / n)
        inicio += n

    return promedios


def promedio_por_grupos_bis(lista, tamanos):
    """
    Calcula promedio y desviación estándar muestral para grupos
    de distinto tamaño, devolviendo ufloats.
    """
    prom = []
    inicio = 0

    for n in tamanos:
        grupo = np.array(lista[inicio:inicio+n])

        media = np.mean(grupo)
        desvio = np.std(grupo, ddof=1)

        prom.append(ufloat(media, desvio))

        inicio += n

    return prom


# Cantidad de mediciones correspondientes a cada campo
tamanos = [3, 4, 4]

SAR_prom = promedio_por_grupos(SAR_M15, tamanos)
tau_prom = promedio_por_grupos(tau_M15, tamanos)
Hc_prom  = promedio_por_grupos(Hc_M15, tamanos)
WR_prom  = promedio_por_grupos_bis(rates_M15, tamanos)

campos = ['H1', 'H2', 'H3']

print(f'Muestra = {nombre_M15}')
print(f'Concentracion = {conc_M15:.1f} g/L\n')

for campo, sar, tau, hc, wr in zip(campos, SAR_prom, tau_prom, Hc_prom, WR_prom):

    print(f'{campo}:')
    print(f'  ESAR = {sar:.3uS} W/g')
    print(f'  tau  = {tau:.1uS} ns')
    print(f'  Hc   = {hc:.1uS} kA/m')
    print(f'  WR   = {wr:.1uS} °C/s')


#%% Idem para Fe
print(f'Muestra = {nombre_Fe}')
print(f'Concentracion = {conc_Fe:.1f} g/L')
SAR_prom = promedio_por_grupos(SAR_Fe, tamanos)
tau_prom = promedio_por_grupos(tau_Fe, tamanos)
Hc_prom  = promedio_por_grupos(Hc_Fe, tamanos)
WR_prom  = promedio_por_grupos_bis(rates_Fe, tamanos)

for campo, sar, tau, hc, wr in zip(campos, SAR_prom, tau_prom, Hc_prom, WR_prom):    
    print(f'{campo}:')
    print(f'  ESAR = {sar:.3uS} W/g')
    print(f'  tau  = {tau:.3uS} ns')
    print(f'  Hc   = {hc:.3uS} kA/m')
    print(f'  WR   = {wr:.1uS} °C/s')
#%%
#%% Guardar resumen de resultados

# Conversión Idc -> H0
H0_dict = {
    '050dA': 20,
    '075dA': 28,
    '100dA': 38,
    '125dA': 47,
    '150dA': 57,
    '152dA': 58,}

muestras = [{'nombre': nombre_M15,
        'conc': conc_M15,
        'resultados': resultados_M15,
        'SAR': SAR_M15,
        'tau': tau_M15,
        'Hc': Hc_M15,
        'WR': rates_M15,
        'archivo': 'Resumen_resultados_M15_260630.txt'
    },
    {
        'nombre': nombre_Fe,
        'conc': conc_Fe,
        'resultados': resultados_Fe,
        'SAR': SAR_Fe,
        'tau': tau_Fe,
        'Hc': Hc_Fe,
        'WR': rates_Fe,
        'archivo': 'Resumen_resultados_Fe_260630.txt'
    }
]

for muestra in muestras:

    with open(muestra['archivo'], 'w', encoding='utf-8') as f:

        f.write(f'Muestra = {muestra["nombre"]}\n')
        f.write(f'Concentracion = {muestra["conc"]:.1f} g/L\n\n')

        f.write(f'{"Medición":<9}{"H0 (kA/m)":<11}{"ESAR (W/g)":<18}{"tau (ns)":<18}{"Hc (kA/m)":<18}{"WR (°C/s)":<18}\n')
        f.write('-'*90 + '\n')

        for i, (res, sar, tau, hc,wr) in enumerate(
                zip(muestra['resultados'],
                    muestra['SAR'],
                    muestra['tau'],
                    muestra['Hc'],
                    muestra['WR']),
                start=1):

            # Buscar el H0 correspondiente
            H0 = np.nan
            for key, value in H0_dict.items():
                if key in res:
                    H0 = value
                    break

            sar_str = f'{sar:.3uS}'
            tau_str = f'{tau:.3uS}'
            hc_str  = f'{hc:.3uS}'
            wr_str  = f'{wr:.2f}'

            f.write(f'{i:<9}{H0:<11.0f}{sar_str:<18}{tau_str:<18}{hc_str:<18}{wr_str:<18}\n')

    print(f'Se guardó: {muestra["archivo"]}')
# %%
