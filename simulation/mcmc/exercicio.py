import gibbs
import metros

### AMOSTRADOR DE GIBBS
valores = {1000:gibbs.gibbs_normal_bi(N=1000) , 10000:gibbs.gibbs_normal_bi(N=10000)}
gibbs.faz_plots(valores[1000])
gibbs.faz_plots(valores[10000])

### METROPOLIS PARA NORMAL ###

### N = 1000 SIGMA = 1
resultado_1000_1 = metros.metro_Normal(chute=[1], N=1000, sigma_aux=1)
metros.faz_plot(resultado_1000_1["valores"])
print("Taxa de Aceitação para 1000 amostras Sigma=1: " + str(resultado_1000_1["taxa"]))

### N = 1000 SIGMA = 5
resultado_1000_5 = metros.metro_Normal(chute=[1], N=1000, sigma_aux=5)
metros.faz_plot(resultado_1000_5["valores"])
print("Taxa de Aceitação para 1000 amostras Sigma=5: " + str(resultado_1000_5["taxa"]))

### N = 10000 SIGMA = 1
resultado_10000_1 = metros.metro_Normal(chute=[1], N=10000, sigma_aux=1)
metros.faz_plot(resultado_10000_1["valores"])
print("Taxa de Aceitação para 10000 amostras Sigma=1: " + str(resultado_10000_1["taxa"]))

### N = 10000 SIGMA = 5
resultado_10000_5 = metros.metro_Normal(chute=[1], N=10000, sigma_aux=5)
metros.faz_plot(resultado_10000_5["valores"])
print("Taxa de Aceitação para 10000 amostras Sigma=5: " + str(resultado_10000_5["taxa"]))


### METROPOLIS PARA EXPO E POISSON

### N = 1000
resultado_1000 = metros.metro_exp_poison()
metros.faz_plot(resultado_1000["valores"])
print("Taxa de Aceitação para 1000: " + str(resultado_1000["taxa"]))

### N = 10000
resultado_10000 = metros.metro_exp_poison(N=10000)
metros.faz_plot(resultado_10000["valores"])
print("Taxa de Aceitação para 10000 amostras: " + str(resultado_10000["taxa"]))
