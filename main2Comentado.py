import flet as ft
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
from flet.matplotlib_chart import MatplotlibChart
import networkx as nx
from scipy.optimize import minimize
import pandas as pd
import tequila as tq
import threading

# Configuramos matplotlib para que use el backend 'svg'
matplotlib.use('svg')

# ============================
# CONSTANTES DE ESTILO
# ============================
BACKGROUND_COLOR = "#FFFAEC"
CARD_COLOR = "#F5ECD5"
PRIMARY_COLOR = "#578E7E"
TEXT_COLOR = "#3D3D3D"

# ============================
# FUNCIÓN PRINCIPAL DE LA APLICACIÓN
# ============================
def main(page: ft.Page):
    # Configuración inicial de la página Flet
    page.title = "Planck"  # Título de la aplicación
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.bgcolor = BACKGROUND_COLOR

    # Variables globales para almacenar datos y la cantidad de shots disponibles
    global data_scaled, labels, shotsFree
    data_scaled = None  # Datos del CSV ya escalados
    labels = None       # Etiquetas (clases) extraídas del CSV
    shotsFree = 100000  # Número total de shots disponibles para las simulaciones cuánticas

    # ============================
    # FUNCIONES PARA CARGAR Y PROCESAR DATOS
    # ============================

    # Función para invocar el selector de archivos (solo se permite un archivo)
    def load_csv(e):
        file_picker.pick_files(allow_multiple=False)
    
    # Función que procesa el archivo CSV seleccionado
    def process_csv(e):
        global data_scaled, labels
        
        if e.files:  # Si se ha seleccionado algún archivo
            file_path = e.files[0].path  # Ruta del archivo seleccionado
            df = pd.read_csv(file_path)  # Leer el CSV en un DataFrame de pandas

            try:
                # Se asume que la última columna corresponde a la etiqueta y las demás a las características
                labels = df.iloc[:, -1].values
                data = df.iloc[:, :-1].values

                # Escalar los datos para normalizarlos usando StandardScaler
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)

                # Actualizar la vista actual para mostrar la vista de configuración
                page.views.clear()
                page.views.append(configuration_view())
                page.update()

                # Mostrar un mensaje de éxito utilizando un SnackBar
                snack_bar = ft.SnackBar(ft.Text("Archivo CSV cargado correctamente."))
                page.overlay.append(snack_bar)
                snack_bar.open = True
                page.update()
            except Exception as ex:
                # En caso de error al procesar el CSV, se muestra un mensaje de error
                snack_bar = ft.SnackBar(ft.Text(f"Error al procesar el CSV: {ex}"))
                page.overlay.append(snack_bar)
                snack_bar.open = True
                page.update()
        else:
            # Mensaje si no se seleccionó ningún archivo
            snack_bar = ft.SnackBar(ft.Text("No se seleccionó ningún archivo."))
            page.overlay.append(snack_bar)
            snack_bar.open = True
            page.update() 

    # Configuración del file_picker, que usará la función process_csv al obtener resultados
    file_picker = ft.FilePicker(on_result=process_csv)
    page.overlay.append(file_picker)

    # ============================
    # VISTA DE LOGIN
    # ============================
    # Función para validar las credenciales del usuario
    def validate_login(e):
        if username_field.value == "Planck" and password_field.value == "Planck123":
            # Si las credenciales son correctas, se muestra la vista de configuración
            page.views.clear()
            page.views.append(configuration_view())
            page.update()
        else:
            # Si las credenciales son incorrectas, se muestra un mensaje de error
            error_message.value = "Credenciales incorrectas."
            error_message.update()

    # Función para reiniciar la ejecución y volver a la vista de configuración
    def execute_again(e):
        page.views.clear()
        page.views.append(configuration_view())
        page.update()

    # Definición de campos y controles para el formulario de login
    username_field = ft.TextField(label="Usuario o email", width=300, bgcolor=TEXT_COLOR)
    password_field = ft.TextField(label="Contraseña", password=True, width=300, bgcolor=TEXT_COLOR)
    error_message = ft.Text(value="", color="red")

    # Vista de login organizada en una columna centrada
    login_view = ft.Column(
        controls=[
            ft.Container(
                content=ft.Image(src="logo.png", width=150, height=150),
                alignment=ft.alignment.center,
            ),
            ft.Text("Planck", size=30, weight=ft.FontWeight.BOLD, color=TEXT_COLOR),
            username_field,
            password_field,
            ft.ElevatedButton("Iniciar sesión", on_click=validate_login, bgcolor=PRIMARY_COLOR, color="white"),
            error_message,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )
    
    # ============================
    # GENERACIÓN DE GRÁFICOS
    # ============================

    # Función para generar un gráfico de dispersión (scatter plot) con los datos cargados
    def generate_plot():
        """Genera un scatter plot de los datos cargados."""
        if data_scaled is None or labels is None:
            return ft.Text("Cargue un archivo CSV para visualizar los datos.", weight=ft.FontWeight.BOLD, color=TEXT_COLOR)
        
        # Crear la figura y el eje usando matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='viridis')
        ax.set_xlabel('Característica 1')
        ax.set_ylabel('Característica 2')
        ax.set_title('Datos cargados del CSV')
        fig.colorbar(scatter, label='Clases')

        return ft.Container(
            content=MatplotlibChart(fig, expand=True),
            width=600,
            height=400,
            alignment=ft.alignment.center
        )

    # Función para generar el gráfico de resultados del QSVM
    def generate_qsvm_plot():
        # Crear figura y eje
        fig, ax = plt.subplots(figsize=(6, 4))

        # Graficar los datos de prueba con las clases reales
        scatter_real = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, label='Clases reales', cmap='viridis')
        # Graficar las predicciones del QSVM con un marcador diferente ('x')
        scatter_pred = ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_pred,
            marker='x',
            label='Predicciones QSVM',
            alpha=0.5,
            cmap='cool',
        )

        # Configuración de etiquetas y título
        ax.set_xlabel('Característica 1')
        ax.set_ylabel('Característica 2')
        ax.set_title('Clasificación QSVM (con kernel cuántico estimado)')
        ax.legend()

        # Añadir barra de color para referenciar las clases reales
        fig.colorbar(scatter_real, ax=ax, label='Clases reales')

        return ft.Container(
            content=MatplotlibChart(fig, expand=True),
            width=600,
            height=400,
            alignment=ft.alignment.center
        )

    # Función para generar el gráfico de la reducción de dimensiones (QPCA)
    def generate_qpca_plot():
        # Crear figura y eje (gráfico unidimensional)
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Graficar la componente principal reducida en el eje X y en Y se colocan ceros
        scatter = ax.scatter(
            reduced_data,
            np.zeros_like(reduced_data),
            c=labels,
            cmap='viridis'
        )

        # Configurar el gráfico: etiqueta del eje X y quitar etiquetas del eje Y
        ax.set_xlabel('Componente Principal 1')
        ax.set_yticks([])  # Eliminar valores en eje Y
        ax.set_title('Datos Iris después de qPCA (1 Componente Principal)')

        # Agregar una barra de color que indique las clases
        fig.colorbar(scatter, ax=ax, label='Clases')

        return ft.Container(
            content=MatplotlibChart(fig, expand=True),
            width=600,
            height=400,
            alignment=ft.alignment.center
        )

    # ============================
    # VISTA DE CARGA
    # ============================
    # Vista intermedia que se muestra mientras se ejecutan tareas cuánticas pesadas
    def loading_view():
        return ft.View(
            route="/loading",
            controls=[
                ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.ProgressRing(color=PRIMARY_COLOR, scale=3),  # Indicador de carga
                            ft.Text("\nProcesando tarea cuántica...", size=25, weight=ft.FontWeight.BOLD, color=TEXT_COLOR),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER
                    ),
                    bgcolor=BACKGROUND_COLOR,
                    expand=True,
                    alignment=ft.alignment.center
                )
            ],
            bgcolor=BACKGROUND_COLOR
        )
        
    # ============================
    # VISTA DE CONFIGURACIÓN
    # ============================
    # Vista donde se seleccionan los parámetros del circuito cuántico y se muestran los datos cargados
    def configuration_view():
        # Dropdown para seleccionar el tipo de circuito ansatz a utilizar
        dropdown = ft.Dropdown(
            label="Tipo de circuito ansatz",
            options=[
                ft.dropdown.Option("Optimización de Parametros (QAOA)"),
                ft.dropdown.Option("Reducción de Datos (QSVM)"),
                ft.dropdown.Option("Reducción de Dimensiones (QPCA)"),
            ],
            width=300,
            bgcolor=TEXT_COLOR
        )
        # Campo para ingresar la cantidad de shots (repeticiones de la simulación)
        repeticiones_field = ft.TextField(label="Shots", width=300, bgcolor=TEXT_COLOR)
        
        # Función que se ejecuta al hacer clic en "Ejecutar"
        def execute_quantum_task(e):
            global shotsFree
            # Se obtiene el tipo de circuito seleccionado y el número de shots ingresado
            circuito_ansatz = dropdown.value
            cantidad_qubits = 2  # Se fija en 2 qubits para este ejemplo
            shots = repeticiones_field.value
            shotsFree = shotsFree - int(shots)  # Descontar los shots utilizados

            # Cambiar a la vista de carga mientras se ejecuta la tarea cuántica
            page.views.append(loading_view())
            page.go("/loading")
            page.update()

            # Ejecutar la tarea cuántica en un hilo separado para no bloquear la interfaz
            def run_task():
                run_quantum_pipeline(circuito_ansatz, cantidad_qubits)
                
                # Al finalizar la tarea, se actualiza la vista de resultados
                page.views.pop()  # Eliminar la vista de carga
                page.views.append(result_view(circuito_ansatz, cantidad_qubits, shots))
                page.update()

            threading.Thread(target=run_task).start()
        
        # Retornar la vista de configuración con los controles dispuestos en un contenedor
        return ft.Container(
            content=ft.Row(
                controls=[
                    ft.Column(
                        controls=[
                            ft.Text("Configuración", size=30, weight=ft.FontWeight.BOLD, color=PRIMARY_COLOR),
                            ft.Text(f"SHOTS AVAILABLE: {shotsFree}", size=20, weight=ft.FontWeight.BOLD, color=TEXT_COLOR),
                            dropdown,
                            repeticiones_field,
                            ft.ElevatedButton("Ejecutar", on_click=execute_quantum_task, bgcolor=PRIMARY_COLOR, color="white"),
                            ft.ElevatedButton("Cargar Datos (CSV)", on_click=load_csv, bgcolor=PRIMARY_COLOR, color="white"),
                            generate_plot(),  # Se muestra el gráfico de los datos cargados
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        expand=True
                    )
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True
            ),
            bgcolor=BACKGROUND_COLOR,
            expand=True
        )

    # ============================
    # VISTA DE RESULTADOS
    # ============================
    # Vista que muestra los resultados obtenidos, variando según el circuito ansatz seleccionado
    def result_view(circuito_ansatz, cantidad_qubits, shots):
        view = None
        
        # Caso QSVM: Resultados de clasificación cuántica
        if(circuito_ansatz == 'Reducción de Datos (QSVM)'):
            view = ft.Column(
                        controls=[
                            ft.Text("Resultados", size=30, weight=ft.FontWeight.BOLD, color=PRIMARY_COLOR),
                            ft.Text(f"El dataset ha sido reducido con exito utilizando {cantidad_qubits} qubits y {shots} shots.", weight=ft.FontWeight.BOLD, color=TEXT_COLOR),
                            ft.Text("La precisión del modelo cuántico ha sido calculada.", weight=ft.FontWeight.BOLD, color=TEXT_COLOR),
                            ft.Text(f"Precisión: {quantum_accuracy:.2f}%", weight=ft.FontWeight.BOLD, color=TEXT_COLOR),
                            generate_qsvm_plot(),
                            ft.ElevatedButton("Finalizar", on_click=lambda e: page.window_close(), bgcolor=PRIMARY_COLOR, color="white"),
                            ft.ElevatedButton("Ejecutar De Nuevo", on_click=execute_again, bgcolor=PRIMARY_COLOR, color="white"),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        expand=True
                    )
            
        # Caso QPCA: Resultados de la reducción de dimensiones cuántica
        if(circuito_ansatz == 'Reducción de Dimensiones (QPCA)'):
            view = ft.Column(
                        controls=[
                            ft.Text("Resultados", size=30, weight=ft.FontWeight.BOLD, color=PRIMARY_COLOR),
                            ft.Text(f"Se redujo las dimensiones del dataset con exito utilizando {cantidad_qubits} qubits y {shots} shots.", weight=ft.FontWeight.BOLD, color=TEXT_COLOR),
                            generate_qpca_plot(),
                            ft.ElevatedButton("Finalizar", on_click=lambda e: page.window_close(), bgcolor=PRIMARY_COLOR, color="white"),
                            ft.ElevatedButton("Ejecutar De Nuevo", on_click=execute_again, bgcolor=PRIMARY_COLOR, color="white"),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        expand=True
                    )
            
        # Caso QAOA: Resultados de la optimización de parámetros para el problema Max-Cut
        if(circuito_ansatz == 'Optimización de Parametros (QAOA)'):
            view = ft.Column(
                        controls=[
                            ft.Text("Resultados", size=30, weight=ft.FontWeight.BOLD, color=PRIMARY_COLOR),
                            ft.Text(f"La optimizacion de parametros se ejecuto exitosamente utilizando {cantidad_qubits} qubits y {shots} shots.", weight=ft.FontWeight.BOLD, color=TEXT_COLOR),
                            ft.Text(f"Los Parámetros óptimos son: {params} y la Energía mínima encontrada es: {energy}", weight=ft.FontWeight.BOLD, color=TEXT_COLOR),
                            ft.ElevatedButton("Finalizar", on_click=lambda e: page.window_close(), bgcolor=PRIMARY_COLOR, color="white"),
                            ft.ElevatedButton("Ejecutar De Nuevo", on_click=execute_again, bgcolor=PRIMARY_COLOR, color="white"),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        expand=True
                    )
        
        return ft.Container(
                    content=ft.Row(
                        controls=[
                            view
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        expand=True
                    ),
                    bgcolor=BACKGROUND_COLOR,
                    expand=True
                )

    # ============================
    # PIPELINE CUÁNTICO
    # ============================
    # Función que ejecuta la tarea cuántica seleccionada (QSVM, QPCA o QAOA)
    def run_quantum_pipeline(circuito_ansatz, cantidad_qubits = 2):
        # Declarar variables globales que serán utilizadas en las vistas de resultados
        global quantum_accuracy  # Precisión de la clasificación QSVM
        global X_train, X_test, y_train, y_test, y_pred  # Variables para el conjunto de datos en QSVM
        global reduced_data  # Datos reducidos obtenidos en QPCA
        global params, energy  # Parámetros óptimos y energía mínima para QAOA
        
        # ----------------------------
        # CASO QSVM: Reducción de Datos para Clasificación
        # ----------------------------
        if(circuito_ansatz == 'Reducción de Datos (QSVM)'):
            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)

            # Función que define el mapa de características cuántico.
            # Se construye un circuito que para cada qubit aplica:
            #   - Una puerta Hadamard para crear superposición.
            #   - Una puerta Ry con el ángulo correspondiente al dato.
            def feature_map(num_qubits, data):
                circuit = tq.QCircuit()
                for i in range(num_qubits):
                    circuit += tq.gates.H(target=i)
                    circuit += tq.gates.Ry(angle=data[i], target=i)
                return circuit

            n_qubits = int(cantidad_qubits)

            # Función que estima el kernel cuántico: se calcula el producto interno (overlap) entre estados
            def quantum_kernel_estimation(x1, x2):
                kernel_matrix = np.zeros((len(x1), len(x2)))
                for i, xi in enumerate(x1):
                    for j, xj in enumerate(x2):
                        # Construir el circuito para cada muestra usando el feature map
                        circuit_i = feature_map(n_qubits, xi)
                        circuit_j = feature_map(n_qubits, xj)
                        # Simular los circuitos para obtener los estados cuánticos correspondientes
                        state_i = tq.simulate(circuit_i, backend="qulacs")
                        state_j = tq.simulate(circuit_j, backend="qulacs")
                        # Calcular el overlap entre los estados y elevar al cuadrado su valor absoluto
                        kernel_matrix[i, j] = np.abs(state_i.inner(state_j)) ** 2
                return kernel_matrix

            # Entrenar un clasificador SVM clásico utilizando el kernel precomputado basado en el kernel cuántico
            svc = SVC(kernel='precomputed')
            kernel_matrix_train = quantum_kernel_estimation(X_train, X_train)
            svc.fit(kernel_matrix_train, y_train)

            # Evaluar el modelo en el conjunto de prueba
            kernel_matrix_test = quantum_kernel_estimation(X_test, X_train)
            y_pred = svc.predict(kernel_matrix_test)

            # Calcular la precisión del modelo
            quantum_accuracy = accuracy_score(y_test, y_pred) * 100

        # ----------------------------
        # CASO QPCA: Reducción de Dimensiones
        # ----------------------------
        if circuito_ansatz == 'Reducción de Dimensiones (QPCA)':
            # Función para transformar datos clásicos a un formato cuántico (normalizando cada vector)
            def classical_to_quantum(data):
                n_features = data.shape[1]
                quantum_data = []
                for row in data:
                    normalized_row = row / np.linalg.norm(row)
                    quantum_data.append(normalized_row)
                return np.array(quantum_data)

            quantum_data = classical_to_quantum(data_scaled)

            # Función que crea un circuito cuántico para cada muestra.
            # Se aplica una puerta Ry parametrizada por cada valor del dato normalizado.
            def create_circuit(data, n_qubits):
                circuit = tq.QCircuit()
                for i in range(n_qubits):
                    circuit += tq.gates.Ry(angle=data[i], target=i)
                return circuit

            n_qubits = quantum_data.shape[1]
            # Crear un circuito para cada muestra del dataset
            circuits = [create_circuit(row, n_qubits) for row in quantum_data]

            # Calcular valores de expectativa para cada circuito simulando su evolución
            expectation_values = []
            for circuit in circuits:
                # Simular el circuito usando Tequila con el backend "qulacs"
                state = tq.simulate(circuit, backend="qulacs")
                # Calcular una expectativa (en este ejemplo se usa la auto-simulación)
                expectation = np.real(state.inner(state))
                expectation_values.append(expectation)

            # Convertir la lista de expectativas en una matriz NumPy
            expectation_matrix = np.array(expectation_values)
            
            # Asegurarse de que la matriz tenga dos dimensiones
            if expectation_matrix.ndim == 1:
                expectation_matrix = expectation_matrix.reshape(-1, 1)

            # Aplicar PCA clásico a los valores de expectativa para reducir dimensiones
            n_samples, n_features = expectation_matrix.shape
            n_components = min(2, min(n_samples, n_features))  # Definir número de componentes

            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(expectation_matrix)
        
        # ----------------------------
        # CASO QAOA: Optimización de Parámetros para Max-Cut
        # ----------------------------
        elif(circuito_ansatz == 'Optimización de Parametros (QAOA)'):

            # Función para crear un grafo aleatorio (modelo Erdős-Rényi)
            def create_graph():
                G = nx.erdos_renyi_graph(12, 0.1)
                return G

            # Función para construir el operador de costo (Hamiltoniano) para el problema Max-Cut.
            # Por cada arista se crea un término basado en el operador Pauli Z aplicado en los nodos correspondientes.
            def maxcut_operator(graph, num_qubits):
                pauli_list = []
                coeffs = []

                for i, j in graph.edges():
                    if i >= num_qubits or j >= num_qubits:
                        raise IndexError(f"Índices de los nodos {i} y {j} están fuera del rango de qubits ({num_qubits})")

                    # Crear un string de Pauli: 'Z' en las posiciones de los nodos y 'I' en las demás
                    pauli_string = ['I'] * num_qubits
                    pauli_string[i], pauli_string[j] = 'Z', 'Z'
                    pauli_list.append(''.join(pauli_string))
                    coeffs.append(-1.0)  # Coeficiente negativo para el operador de costo

                return pauli_list, coeffs
            
            # Función que crea el circuito QAOA
            # Se inicia con puertas Hadamard y luego se aplican capas alternadas de operador de costo y operador de mezcla
            def create_qaoa_circuit(num_qubits, params):
                layers = len(params) // 2  # Cada capa tiene dos parámetros: gamma y beta
                circuit = tq.QCircuit()

                # Aplicar puertas Hadamard para crear superposición en todos los qubits
                for qubit in range(num_qubits):
                    circuit += tq.gates.H(target=qubit)

                # Aplicar las capas QAOA
                for i in range(layers):
                    gamma, beta = params[2 * i], params[2 * i + 1]
                    # Aplicar términos del operador de costo mediante secuencias de CNOT y Rz
                    for j, k in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
                        circuit += tq.gates.CNOT(target=j, control=k)
                        circuit += tq.gates.Rz(angle=2 * gamma, target=k)
                        circuit += tq.gates.CNOT(target=j, control=k)
                    # Aplicar operador de mezcla con compuertas Rx en cada qubit
                    for qubit in range(num_qubits):
                        circuit += tq.gates.Rx(angle=2 * beta, target=qubit)

                return circuit
            
            # Función que ejecuta QAOA optimizando la función de costo con un método clásico
            def run_qaoa(graph, p=1):
                num_qubits = graph.number_of_nodes()
                pauli_list, coeffs = maxcut_operator(graph, num_qubits)

                backend = 'qulacs'  # Seleccionamos el backend de simulación cuántica
                estimator = tq.simulate  # Usamos la función de simulación de Tequila

                # Inicialización aleatoria de los parámetros (ángulos) para QAOA
                params = np.random.uniform(0, 2 * np.pi, size=2 * p)
                
                # Función auxiliar para convertir un string de Pauli a un diccionario (requerido por Tequila)
                def pauli_string_to_dict(pauli_string):
                    pauli_dict = {}
                    for idx, p in enumerate(pauli_string):
                        if p != 'I':  # Solo se consideran operadores no identidad
                            pauli_dict[idx] = p.lower()
                    return pauli_dict

                # Función de costo que calcula la energía esperada para un conjunto de parámetros
                def cost_function(params):
                    circuit = create_qaoa_circuit(num_qubits, params)
                    wavefunction = estimator(circuit, backend=backend)
                    
                    energy = 0
                    for pauli_string, coeff in zip(pauli_list, coeffs):
                        pauli_dict = pauli_string_to_dict(pauli_string)
                        pauli_operator = tq.hamiltonian.PauliString(pauli_dict)
                        # Aplicar el operador de Pauli al estado cuántico y calcular la contribución en energía
                        result_wavefunction = wavefunction.apply_paulistring(pauli_operator)
                        energy += coeff * result_wavefunction.inner(other=wavefunction)
                        
                    return energy

                # Optimización clásica de la función de costo utilizando el método COBYLA
                result = minimize(cost_function, params, method="COBYLA", options={"maxiter": 100})
                optimal_params = result.x
                optimal_energy = result.fun

                return optimal_params, optimal_energy
            
            # Crear un grafo aleatorio para el problema Max-Cut
            graph = create_graph()
            print("Grafo:", graph.edges())

            # Ejecutar QAOA con p capas (en este ejemplo p=2)
            p = 2
            params, energy = run_qaoa(graph, p=p)

            print("\nParámetros óptimos:", params)
            print("Energía mínima encontrada:", energy)

    # ============================
    # AGREGAR VISTA INICIAL A LA PÁGINA
    # ============================
    page.add(login_view)

# Ejecutar la aplicación Flet
ft.app(target=main)
