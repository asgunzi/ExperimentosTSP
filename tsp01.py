# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 07:24:05 2025

@author: asgun
"""


import random
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from scipy.optimize import linear_sum_assignment

def gera_base_tsp(n, val_seed =-1):
    base_tsp=[]
    
    if val_seed==-1:
        random.seed()
    else:
        random.seed(val_seed)

    for i in range(n):
        base_tsp.append([i, random.randint(10, 100), random.randint(10, 100)])
        
    return(base_tsp)

def dist_euclid(p0, p1):
    #Atencao. Devido ao formato idx lat long, pega coluna 1 e 2
    return( sqrt((p0[1]-p1[1])**2 +(p0[2]-p1[2])**2))

def calc_distancias(base, rota):
    dist =0
    for i in range(len(rota)-1):
        idx1 = rota[i]
        idx2 = rota[i+1]
        dist += dist_euclid(base[idx1], base[idx2])

    #Fecha o ultimo
    idx1 = rota[len(rota)-1]
    idx2 = rota[0]
    dist += dist_euclid(base[idx1], base[idx2])
    return(dist)

def plotaPontos(base,titulo):
    coordX = []
    coordY = []

    for n in range(len(base)):
        coordX.append(base[n][1])
        coordY.append(base[n][2])
    
    plt.scatter(coordX, coordY)
    plt.title(titulo)
    plt.show()


def plotaRota(base,rota, titulo):
    #Plota base
    coordX = []
    coordY = []
    for n in range(len(base)):
        coordX.append(base[n][1])
        coordY.append(base[n][2])
    
    plt.scatter(coordX, coordY)

    #Todos os pontos exceto o último
    for n in range(len(rota)-1):
        idxn = rota[n]
        idxn2 =rota[n+1]
        plt.plot([base[idxn][1], base[idxn2][1]], [base[idxn][2], base[idxn2][2]], color = 'black', linestyle=':')


    #Fecha o último
    idxn = rota[len(rota)-1]
    idxn2 =rota[0]
    plt.plot([base[idxn][1], base[idxn2][1]], [base[idxn][2], base[idxn2][2]], color = 'black', linestyle=':')

    plt.title(titulo)
    plt.show()
        
    

def greedy(base):
    #Começa do primeiro e vai acrescentando a menor distancia
    idxOrigem = 0
    out_rota =[idxOrigem]
    dist_total =0
    dist =0
    
    n= len(base)
    
    for _ in range(n-1): #Para cada cidade origem
        menorDist = 1000000
        idxDestino =0
        #Verifica todas distancias para destino
        for p in range(n):
            if p not in out_rota: #Não deve ter sido visitada
                if p != idxOrigem: #Deve ser diferente de si mesmo
                    dist = dist_euclid(base1[idxOrigem],base1[p])
                    if dist< menorDist:
                        menorDist = dist
                        idxDestino = p
        
        out_rota.append(idxDestino)
        idxOrigem = idxDestino #Para próxima iteração
        dist_total  +=menorDist


    #Acrescenta distância último - origem
    dist_total += dist_euclid(base1[out_rota[-1]],base1[0])
        
    return(out_rota,dist_total)
    

def two_opt_adjacente(base, rota):
    n = len(rota)
    nova_rota = rota

    for i in range(n-3):
        ganho = dist_euclid(base[nova_rota[i]], base[nova_rota [i+1]])
        ganho += dist_euclid(base[nova_rota [i+2]], base[nova_rota [i+3]])
        ganho -= dist_euclid(base[nova_rota [i]], base[nova_rota [i+2]])
        ganho -= dist_euclid(base[nova_rota [i+1]], base[nova_rota [i+3]])
        
        if ganho>0:
            nova_rota = nova_rota[:i+1] + [nova_rota[i+2], nova_rota[i+1]] + nova_rota[i+3:]

    return nova_rota
            
    
def two_opt_swap(base, rota, i, k):
    nova_rota =rota
    
    ganho = dist_euclid(base[rota[i]], base[rota[i+1]])
    ganho += dist_euclid(base[rota[k]], base[rota[k+1]])
    ganho -= dist_euclid(base[rota[i]], base[rota[k]])
    ganho -= dist_euclid(base[rota[i+1]], base[rota[k+1]])
    
    if ganho>0:
        nova_rota = rota[:i+1] + rota[i+1:k+1][::-1] + rota[k+1:]

    return nova_rota

def two_opt(base, rota):
    #Aplica algoritmo two opt
    for i in range(1, len(rota) - 2):
        for k in range(i + 2, len(rota)-1):
            rota = two_opt_swap(base, rota, i, k)
    return rota


#=================

def encontra_farthest_ini(base):
    #Encontra o ponto mais distante da rota atual
    n = len(base)
    dist_base =0
    
    #Testa todos os pares e verifica maior dist
    for i in range(n-1):
        for j in range(i+1,n):
           calc = dist_euclid(base[i],base[j])
           if calc > dist_base:
               dist_base = calc
               idxi = i
               idxj = j
    
    out_pontos =[idxi, idxj]
                   
    return(out_pontos)    

def farthest(base,rota_atual):
    n = len(base)
    dist_maior = 0
    candidatos = []
    for j in range(n):
        if j not in rota_atual:
            candidatos.append(j)
            
    
    #Testa todos os pontos fora da rota_atual e verifica maior dist minima
    for j in candidatos:
        dist_min =10000
        for i in rota_atual:
            calc = dist_euclid(base[i],base[j])
            if calc > dist_min:
                dist_min = calc
        
        if dist_min > dist_maior:
            dist_maior = dist_min
            idxj = j
    
        
    #Retorna ponto minimo mais distante
    return(idxj)

def insercao2(base, rota_atual, idx):
    n = len(rota_atual)

    diff_min = 10000000
    
    #Para todos menos o último
    for i in range(n-1):
        rota_cand =rota_atual[:i+1] +[idx] +rota_atual[i+1:]
        dist_swap = calc_distancias(base, rota_cand)
        if (dist_swap)<diff_min:
           diff_min = dist_swap
           idx_ref = i
    rota_nova =rota_atual[:idx_ref +1] +[idx] +rota_atual[idx_ref +1:]
    
    #Para o último ponto
    rota_cand =rota_atual +[idx]
    dist_swap = calc_distancias(base, rota_cand)
    if (dist_swap)<diff_min:
        diff_min = dist_swap
        rota_nova =rota_atual +[idx]

    return(rota_nova)

        


def insercao(base, rota_atual, idx):
    #Testa qual o ponto de melhor insercao do ponto idx
    distMin = 1000000
    idx_ref = -1
    n = len(rota_atual)
    
    #Testa todas as posicoes menos última
    for i in range(n-1):
        calc = dist_euclid(base[rota_atual[i]], base[idx]) 
        calc += dist_euclid(base[idx], base[rota_atual[i+1]])

        if calc<distMin:
            distMin = calc
            idx_ref = i
    
    #Testa última posicao
    calc = dist_euclid(base[rota_atual[n-1]], base[idx]) + dist_euclid(base[idx], base[rota_atual[0]])
    if calc<distMin:
        distMin = calc
        idx_ref = n-1

    rota_nova = rota_atual[:idx_ref+1] +[idx] +rota_atual[idx_ref+1:]

    return(rota_nova)    

def encontra_farthest(base):
    n = len(base)
    
    #Encontra pontos iniciais
    rota_atual = encontra_farthest_ini(base)
    
    for k in range(n-2):
        #Encontra o ponto mais distante
        idx = farthest(base,rota_atual)
        
        #Encontra o melhor ponto de inserção e Insere na rota
        #rota_atual = insercao(base, rota_atual, idx)
        rota_atual = insercao2(base, rota_atual, idx)
        
    return(rota_atual)


def prim(base):
    #Algoritmo de Prim para encontrar Minimum Spanning Tree
    n = len(base)
    
    #Ponto inicial
    vertices ={0}
    ramos =[]
    
    for i in range(n-1):
        distMin = 10000000
        idxOrigem = -1
        idxDestino = -1
        
        #Verifica qual o ponto mais próximo da árvore até o momento
        for origem in vertices:
            for k in range(n):
                if k not in vertices:
                    dist= dist_euclid(base[origem], base[k])
                    if dist < distMin:
                        distMin = dist
                        idxOrigem = origem
                        idxDestino = k
        #Insere no set de vértices e no de ramos
        vertices.add(idxDestino)
        ramos.append([idxOrigem,idxDestino])
                
    return(vertices, ramos)
        
def plotaRotaPrim(base,ramos):
    n = len(base)

    #Plota resultado
    for i in range(n):
        plt.scatter(base[i][1],base[i][2], color = 'black')
    
    for r in ramos:
        plt.plot([base[r[0]][1],base[r[1]][1]], [base[r[0]][2],base[r[1]][2]], color = 'black', linestyle=':')

    plt.title("Minimum Spanning Tree - Prim")
    plt.show()

        
def double_tree(vert,ramos):
    #Faz double tree a partir dos vértices e ramos do Minimum Spanning Tree
    #Pega a primeira cidade não visitada, a partir do MST
    
    n = len(vert)
    out_rota= ramos[0]
    ramos.remove(ramos[0])
    pont_ramos = 1 #Aponta para posicao onde o vértice está
    
    for _ in range(n-1):
        #Para todos os vértices não visitados

        isFound = False
        while(not isFound):
            vert_alvo = out_rota[pont_ramos]

            for r in ramos:
                if vert_alvo in r:
                    if r[0] not in out_rota:
                        out_rota.append(r[0]) #Adiciona vértice novo
                        ramos.remove(r) #Remove ramo atual
                        pont_ramos = len(out_rota)-1
                        isFound = True
                        break
                    elif r[1] not in out_rota:
                        out_rota.append(r[1]) #Adiciona vértice novo
                        ramos.remove(r) #Remove ramo atual
                        isFound = True
                        pont_ramos = len(out_rota)-1
                        break
                
            if not isFound:
                pont_ramos -=1
                if pont_ramos ==-1:
                    isFound = True
    return(out_rota)

def conta_impar(mst_ref):
    #Dado um minimum spanning tree, conta quais vértices sao ímpares
    n = len(mst_ref)+1
    out_vert =[]
    for v in range(n):
        count=0
        for edge in mst_ref:
            if v in edge:
                count+=1
        if count % 2 == 1:
            out_vert.append(v)
    return(out_vert)
    

def match_simples(base_ref, vert_ref):
    #Faz um match simples entre valores de vert_ref
    #Pega o mais próximo
    out_ramos=[]
    n = len(vert_ref)
    vec_proibido = np.zeros(n, dtype = int)
    
    
    for i in range(n-1):
        distMin = 100000
        idxj = -1
        for j in range(i+1,n):
            v0 = vert_ref[i]
            v1 = vert_ref[j]
            if vec_proibido[j] ==0: #Ou seja, não foi usado antes
                dist_ref = dist_euclid(base_ref[v0], base_ref[v1])
                if dist_ref<distMin:
                   distMin = dist_ref
                   idxj = j        
        if idxj > -1:
            out_ramos.append([vert_ref[i],vert_ref[idxj]])
            vec_proibido[i]=1
            vec_proibido[idxj]=1
    
    return(out_ramos)

def match_linear(base_ref, vert_ref):
    
    cost =[]
    
    for v in vert_ref:
        arr2 = []
        for u in vert_ref:
            if u == v:
                arr2.append(100000)
            else:
                arr2.append(dist_euclid(base_ref[u], base_ref[v]))
        cost.append(arr2)
    
    row_idx, col_idx = linear_sum_assignment(cost)
    out_ramos=[]
    
    for r in row_idx:
        for c in col_idx:
            out_ramos.append([vert_ref[r],vert_ref[c]])

    return(out_ramos)

def christophides(base):

    #Algoritmo de Prim
    vert, ramos2 = prim(base1)

    #Conta ramos impares
    impares = conta_impar(ramos2)
    #print(f'Vértices ímpares {impares}')

    #Match ramos impares        
    #ramos3 = match_simples(base1, impares)
    ramos3 = match_linear(base1, impares)
    
    #Une ramos
    ramos4 =ramos2+ramos3
    
    #Composicao de rota    
    n = len(base)
    out_rota= ramos4[0]
    ramos4.remove(ramos4[0])
    pont_ramos = 1 #Aponta para posicao onde o vértice está
    
    for _ in range(n-1):
        #Para todos os vértices não visitados

        isFound = False
        while(not isFound):
            vert_alvo = out_rota[pont_ramos]

            for r in ramos4:
                if vert_alvo in r:
                    if r[0] not in out_rota:
                        out_rota.append(r[0]) #Adiciona vértice novo
                        ramos4.remove(r) #Remove ramo atual
                        pont_ramos = len(out_rota)-1
                        isFound = True
                        break
                    elif r[1] not in out_rota:
                        out_rota.append(r[1]) #Adiciona vértice novo
                        ramos4.remove(r) #Remove ramo atual
                        isFound = True
                        pont_ramos = len(out_rota)-1
                        break
                
            if not isFound:
                pont_ramos -=1
                if pont_ramos ==-1:
                    isFound = True
    return(out_rota)
    
    
import networkx as nx

def christofides_tsp(graph):
    # Step 1: Create a minimum spanning tree (MST)
    mst = nx.minimum_spanning_tree(graph)

    # Step 2: Find all vertices with an odd degree in the MST
    odd_degree_nodes = [node for node in mst.nodes if mst.degree(node) % 2 != 0]

    # Step 3: Create a subgraph with odd degree vertices and find a minimum-weight perfect matching
    subgraph = graph.subgraph(odd_degree_nodes)
    matching = nx.algorithms.matching.max_weight_matching(subgraph, maxcardinality=True)

    # Step 4: Combine the MST and the matching to form a multigraph
    multigraph = nx.MultiGraph(mst)
    multigraph.add_edges_from(matching)

    # Step 5: Find an Eulerian circuit in the multigraph
    eulerian_circuit = list(nx.eulerian_circuit(multigraph))

    # Step 6: Convert the Eulerian circuit to a Hamiltonian circuit by shortcutting repeated vertices
    hamiltonian_circuit = []
    visited = set()
    for u, v in eulerian_circuit:
        if u not in visited:
            hamiltonian_circuit.append(u)
            visited.add(u)
    hamiltonian_circuit.append(hamiltonian_circuit[0])  # Return to the starting node

    return hamiltonian_circuit


#=================
#Gera base
base1 = gera_base_tsp(25)

'''
#Plota só pontos
plotaPontos(base1, "Traveling Salesman Problem")


#Aleatorio
rota_aleatoria = list(range(len(base1)))
random.shuffle(rota_aleatoria)
plotaRota(base1, rota_aleatoria, "Aleatória")
distancia = calc_distancias(base1, rota_aleatoria)
print(f"Dist aleatória {distancia:.1f}")


#Resolve por algoritmo greedy
rota_greedy, dist_greedy = greedy(base1)
#print(rota_greedy)

plotaRota(base1, rota_greedy, "Greedy")

distancia = calc_distancias(base1, rota_greedy)
print(f"Dist greedy {distancia:.1f}")


rota_two_adj = two_opt_adjacente(base1, rota_greedy)

plotaRota(base1, rota_two_adj , "Two Opt Adjacente")
distancia = calc_distancias(base1, rota_two_adj)
print(f"Dist two opt Adjacente {distancia:.1f}")


#Melhora a solução com o 2-opt
rota_two = two_opt(base1, rota_two_adj)

plotaRota(base1, rota_two, "Two Opt")
distancia = calc_distancias(base1, rota_two)
print(f"Dist two opt {distancia:.1f}")
#print(rota_two)

'''

#Farthest insertion algoritmo
rota_fart = encontra_farthest(base1)
plotaRota(base1, rota_fart , "Farthest")
distancia = calc_distancias(base1, rota_fart)
print(f"Dist farthest {distancia:.1f}")

#Farthest + two opt 
rota_fart_two = two_opt(base1, rota_fart)
plotaRota(base1, rota_fart_two , "Farthest + 2opt")
distancia = calc_distancias(base1, rota_fart_two)
print(f"Dist farthest two {distancia:.1f}")

#Farthest + two opt + two opt adjacente
rota_fart_two_adj = two_opt(base1, rota_fart_two )
plotaRota(base1, rota_fart_two_adj, "Farthest + 2opt + adj")
distancia = calc_distancias(base1, rota_fart_two_adj)
print(f"Dist farthest two + adj {distancia:.1f}")

#Algoritmo de Prim
vert, ramos2 = prim(base1)
plotaRotaPrim(base1, ramos2)

#Algoritmo double tree
rota_double = double_tree(vert,ramos2)

plotaRota(base1, rota_double, "Rota Double MST")
distancia = calc_distancias(base1, rota_double)
print(f"Dist Double MST {distancia:.1f}")


#Christophides
rota_christ = christophides(base1)

plotaRota(base1, rota_christ, "Rota Christophides")
distancia = calc_distancias(base1, rota_christ)
print(f"Dist Christophides {distancia:.1f}")

rota_two_christ = two_opt_adjacente(base1, rota_christ)
plotaRota(base1, rota_two_christ , "Rota Christophides + Two")
distancia = calc_distancias(base1, rota_two_christ)
print(f"Dist Christophides Two {distancia:.1f}")

#Testa algoritmo da IA

#Gera matriz de distancias
arrDist =[]
n1 = len(base1)
for i in range(n1):
    distLin =[]
    for j in range(n1):
        d = dist_euclid(base1[i], base1[j])
        distLin.append(d)
    arrDist.append(distLin)
    arrDist2 = np.array(arrDist)

graph = nx.convert_matrix.from_numpy_array(arrDist2)

# Run Christofides' algorithm
tour = christofides_tsp(graph)

plotaRota(base1, tour, "Rota Christophides AI")
distancia = calc_distancias(base1, tour)
print(f"Dist Christophides AI {distancia:.1f}")

#Algoritmos greedy, insercao, 2opt, 3opt, lin kernigham

