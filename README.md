# Proyecto Final 2025-1: Pong AI
## **CS2013 Programación III** · Informe Final

### **Descripción**

Implementación de una red neuronal multicapa en C++ desde cero para controlar un agente que juega Pong. El sistema aplica aprendizaje por refuerzo con SARSA, arquitectura modular de capas, normalización de entradas, exploración ε-greedy y entrenamiento supervisado. No se utilizó ninguna librería de machine learning externa.

---

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)

---

### Datos generales

* **Tema**: Redes Neuronales para Aprendizaje por Refuerzo
* **Grupo**: `j3ingineers`
* **Integrantes**:

    * Diego Antonio Escajadillo Guerrero - 201910150

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior / MSVC 2019 con C++17
2. **Dependencias**:

    * CMake 3.18+
    * Ninguna otra librería externa

3. **Instalación**:

   ```bash
   git clone https://github.com/usuario/proyecto-pong-ai.git
   cd proyecto-pong-ai
   mkdir build && cd build
   cmake ..
   make
   ```

---

### 1. Investigación teórica

* Historia de las redes neuronales
* Aprendizaje por refuerzo: SARSA vs Q-Learning
* Exploración-explotación: política ε-greedy
* Normalización de entradas y funciones de activación (ReLU)

---

### 2. Diseño e implementación

#### 2.1 Arquitectura modular

* Capas como clases que heredan de `ILayer`
* Optimización mediante `SGD`
* Entrenamiento on-policy con `learnOnPolicy()`

#### 2.2 Organización del proyecto

```
Pong_AI/
├── include/utec/nn/         # Capas, red y optimizadores
├── include/utec/agent/      # Agente SARSA y entorno Pong
├── src/                     # Implementación
├── main.cpp                 # Entrenamiento completo
```

#### 2.3 Casos de prueba

* Test de forward y backward en `Dense`
* Evaluación de convergencia del agente
* Validación del modelo entrenado (`test_agent_env.cpp`)

---

### 3. Ejecución

1. Compilar el proyecto
2. Ejecutar entrenamiento:
   ```bash
   ./Pong_AI
   ```
3. Ejecutar evaluación del modelo:
   ```bash
   ./test_agent_env
   ```
4. Analizar resultados:
    * `pesos.txt`: pesos del modelo
    * `winrate.csv`: desempeño por bloques de entrenamiento

---

### 4. Análisis del rendimiento

* **Episodios entrenados**: 3000
* **Duración aprox.**: ~5 sec
* **Modelo final**: arquitectura 3–16–8–1 con activaciones ReLU
* **Winrate evaluado**: 35–45% promedio sin replay buffer
* **Observaciones**:
    * 🟢 Entrena sin dependencias externas
    * 🔴 Limitado por no usar experiencia acumulada
* **Mejoras futuras**:
    * Implementar Q-Learning off-policy
    * Agregar replay buffer y batch training
    * Exportar métricas y visualizar en Python

* **Video de Demostración**:
 * https://youtu.be/ksVcA2Vv55g
---

### 5. Trabajo en equipo

| Tarea                        | Miembro       | Rol                           |
|-----------------------------|----------------|--------------------------------|
| Investigación teórica       | Diego Escajadillo     | Documentar teoría RL y NNs     |
| Diseño y normalización      |                 | Arquitectura + entradas        |
| Implementación de la red    |                 | Clases Dense, ReLU, NN         |
| Entrenamiento SARSA         |                 | Agente y lógica de aprendizaje |
| Pruebas y winrate.csv       |                 | Registro y visualización       |
| Documentación final         |                    | README, licencias y demo       |

---

### 6. Conclusiones

* Se logró implementar un sistema completo de RL con SARSA desde cero.
* El agente aprende parcialmente, aunque el rendimiento es limitado por falta de memoria.
* Aprendimos a integrar teoría, entrenamiento supervisado y refuerzo.
* El proyecto es una buena base para evolucionar hacia DQN o PPO.

---

### 7. Bibliografía

[1] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, MIT Press, 2018.  
[2] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*, MIT Press, 2016.  
[3] F. Chollet, *Deep Learning with Python*, Manning, 2017.  
[4] D. Silver et al., "Mastering the game of Go with deep neural networks and tree search," *Nature*, vol. 529, 2016.

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.
