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

matplotlib.use('svg')

BACKGROUND_COLOR = "#FFFAEC"
CARD_COLOR = "#F5ECD5"
PRIMARY_COLOR = "#578E7E"
TEXT_COLOR = "#3D3D3D"

def main(page: ft.Page):
    # Configuración inicial de la página
    page.title = "Planck"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.bgcolor = BACKGROUND_COLOR
    global data_scaled, labels, shotsFree
    
    # Variables iniciales
    data_scaled = None
    labels = None
    shotsFree = 100000
    
    # Función para cargar datos desde un archivo CSV.
    def load_csv(e):
        file_picker.pick_files(allow_multiple=False)
    
    # Procesa el archivo CSV seleccionado.
    def process_csv(e):
        global data_scaled, labels
        
        if e.files:
            file_path = e.files[0].path
            df = pd.read_csv(file_path)

            try:
                # Suponiendo que el último atributo es la etiqueta
                labels = df.iloc[:, -1].values
                data = df.iloc[:, :-1].values

                # Escalando los datos
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)

                # Actualizar la gráfica
                page.views.clear()
                page.views.append(configuration_view())
                page.update()

                # Usando Page.overlay.append para mostrar el SnackBar
                snack_bar = ft.SnackBar(ft.Text("Archivo CSV cargado correctamente."))
                page.overlay.append(snack_bar)
                snack_bar.open = True
                page.update()
            except Exception as ex:
                snack_bar = ft.SnackBar(ft.Text(f"Error al procesar el CSV: {ex}"))
                page.overlay.append(snack_bar)
                snack_bar.open = True
                page.update()
        else:
            snack_bar = ft.SnackBar(ft.Text("No se seleccionó ningún archivo."))
            page.overlay.append(snack_bar)
            snack_bar.open = True
            page.update() 

    file_picker = ft.FilePicker(on_result=process_csv)
    page.overlay.append(file_picker)

    # Función de validación de login
    def validate_login(e):
        if username_field.value == "Planck" and password_field.value == "Planck123":
            page.views.clear()
            page.views.append(configuration_view())
            page.update()
        else:
            error_message.value = "Credenciales incorrectas."
            error_message.update()

    def execute_again(e):
        page.views.clear()
        page.views.append(configuration_view())
        page.update()

    # Vista de inicio de sesión
    username_field = ft.TextField(label="Usuario o email", width=300, bgcolor=TEXT_COLOR)
    password_field = ft.TextField(label="Contraseña", password=True, width=300, bgcolor=TEXT_COLOR)
    error_message = ft.Text(value="", color="red")

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
    
    def generate_plot():
        """Genera un scatter plot de los datos cargados."""
        if data_scaled is None or labels is None:
            return ft.Text("Cargue un archivo CSV para visualizar los datos.", weight=ft.FontWeight.BOLD, color=TEXT_COLOR)
        
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

    def generate_qsvm_plot():
        # Crear la figura y los ejes
        fig, ax = plt.subplots(figsize=(6, 4))

        # Visualización de los resultados
        scatter_real = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, label='Clases reales', cmap='viridis')
        scatter_pred = ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_pred,
            marker='x',
            label='Predicciones QSVM',
            alpha=0.5,
            cmap='cool',
        )

        # Etiquetas y título
        ax.set_xlabel('Característica 1')
        ax.set_ylabel('Característica 2')
        ax.set_title('Clasificación QSVM (con kernel cuántico estimado)')
        ax.legend()

        # Agregar una barra de color (opcional)
        fig.colorbar(scatter_real, ax=ax, label='Clases reales')

        # Retornar la gráfica como un control MatplotlibChart de Flet
        return ft.Container(
            content=MatplotlibChart(fig, expand=True),
            width=600,
            height=400,
            alignment=ft.alignment.center
        )

    def generate_qpca_plot():
        # Crear la figura y el eje
        fig, ax = plt.subplots(figsize=(6, 4))  # Dimensiones más alargadas para un gráfico unidimensional

        # Gráfico de dispersión unidimensional
        scatter = ax.scatter(
            reduced_data,
            np.zeros_like(reduced_data),
            c=labels,
            cmap='viridis'
        )

        # Configuración del gráfico
        ax.set_xlabel('Componente Principal 1')
        ax.set_yticks([])  # Eliminar valores en el eje Y porque no tienen significado
        ax.set_title('Datos Iris después de qPCA (1 Componente Principal)')

        # Barra de color para las clases
        fig.colorbar(scatter, ax=ax, label='Clases')

        # Retornar la figura como un control de Flet
        return ft.Container(
            content=MatplotlibChart(fig, expand=True),
            width=600,
            height=400,
            alignment=ft.alignment.center
        )

    def loading_view():
        return ft.View(
            route="/loading",
            controls=[
                ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.ProgressRing(color=PRIMARY_COLOR, scale=3),  # Spinner de carga
                            ft.Text("\nProcesando tarea cuántica...", size=25, weight=ft.FontWeight.BOLD, color=TEXT_COLOR),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,             # Centra verticalmente
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER  # Centra horizontalmente
                    ),
                    bgcolor=BACKGROUND_COLOR,
                    expand=True,                                           # Ocupa todo el espacio disponible
                    alignment=ft.alignment.center                          # Centra el contenido en X e Y
                )
            ],
            bgcolor=BACKGROUND_COLOR
        )
        
    # Vista de configuración
    def configuration_view():
        # Crear los controles de la vista
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
        #qubits_field = ft.TextField(label="Cantidad de qubits", width=300, bgcolor=TEXT_COLOR)
        repeticiones_field = ft.TextField(label="Shots", width=300, bgcolor=TEXT_COLOR)
        
        def execute_quantum_task(e):
            global shotsFree
            # Obtener los valores de los campos
            circuito_ansatz = dropdown.value
            cantidad_qubits = 2
            shots = repeticiones_field.value
            shotsFree = shotsFree - int(shots)
            
            # Cambiar a la vista de carga
            page.views.append(loading_view())
            page.go("/loading")  # Navegar a la vista de carga
            page.update()

            # Ejecutar la tarea pesada en segundo plano
            def run_task():
                run_quantum_pipeline(circuito_ansatz, cantidad_qubits)
                
                # Mostrar los resultados al finalizar
                page.views.pop()  # Eliminar la vista de carga
                page.views.append(result_view(circuito_ansatz, cantidad_qubits, shots))
                page.update()

            threading.Thread(target=run_task).start()
        
        # Retornar la vista con los controles envueltos en un contenedor con color de fondo
        return ft.Container(
            content=ft.Row(
                controls=[
                    ft.Column(
                        controls=[
                            ft.Text("Configuración", size=30, weight=ft.FontWeight.BOLD, color=PRIMARY_COLOR),
                            ft.Text(f"SHOTS AVAILABLE: {shotsFree}", size=20, weight=ft.FontWeight.BOLD, color=TEXT_COLOR),
                            dropdown,
                            #qubits_field,
                            repeticiones_field,
                            ft.ElevatedButton("Ejecutar", on_click=execute_quantum_task, bgcolor=PRIMARY_COLOR, color="white"),
                            ft.ElevatedButton("Cargar Datos (CSV)", on_click=load_csv, bgcolor=PRIMARY_COLOR, color="white"),
                            generate_plot(),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        expand=True  # Expande la columna
                    )
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True  # Expande la fila
            ),
            bgcolor=BACKGROUND_COLOR,
            expand=True  # Expande el contenedor
        )

    # Vista de resultados
    def result_view(circuito_ansatz, cantidad_qubits, shots):
        view = None
        
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
                        expand=True  # Expande la fila
                    ),
                    bgcolor=BACKGROUND_COLOR,
                    expand=True  # Expande el contenedor
                )

    # Función principal de procesamiento cuántico
    def run_quantum_pipeline(circuito_ansatz, cantidad_qubits = 2):
        global quantum_accuracy  # Para acceder desde la vista de resultados
        global X_train, X_test, y_train, y_test, y_pred
        global reduced_data
        global params, energy
        
        if(circuito_ansatz == 'Reducción de Datos (QSVM)'):
            # Dividir los datos en conjunto de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)

            # Definir un mapa de características cuántico (QSVM) con Tequila
            def feature_map(num_qubits, data):
                circuit = tq.QCircuit()
                for i in range(num_qubits):
                    circuit += tq.gates.H(target=i)
                    circuit += tq.gates.Ry(angle=data[i], target=i)
                return circuit

            n_qubits = int(cantidad_qubits)

            # Kernel cuántico estimado usando producto interno
            def quantum_kernel_estimation(x1, x2):
                kernel_matrix = np.zeros((len(x1), len(x2)))
                for i, xi in enumerate(x1):
                    for j, xj in enumerate(x2):
                        circuit_i = feature_map(n_qubits, xi)
                        circuit_j = feature_map(n_qubits, xj)
                        state_i = tq.simulate(circuit_i, backend="qulacs")
                        state_j = tq.simulate(circuit_j, backend="qulacs")
                        kernel_matrix[i, j] = np.abs(state_i.inner(state_j)) ** 2
                return kernel_matrix

            # Entrenar un modelo SVM clásico con el kernel cuántico estimado
            svc = SVC(kernel='precomputed')
            kernel_matrix_train = quantum_kernel_estimation(X_train, X_train)
            svc.fit(kernel_matrix_train, y_train)

            # Evaluar el modelo en los datos de prueba
            kernel_matrix_test = quantum_kernel_estimation(X_test, X_train)
            y_pred = svc.predict(kernel_matrix_test)

            quantum_accuracy = accuracy_score(y_test, y_pred) * 100

        if circuito_ansatz == 'Reducción de Dimensiones (QPCA)':
            # Transformar datos clásicos a un espacio cuántico
            def classical_to_quantum(data):
                n_features = data.shape[1]
                quantum_data = []
                for row in data:
                    # Normalizar cada fila
                    normalized_row = row / np.linalg.norm(row)
                    quantum_data.append(normalized_row)
                return np.array(quantum_data)

            quantum_data = classical_to_quantum(data_scaled)

            # Definir un circuito cuántico para qPCA usando Tequila
            def create_circuit(data, n_qubits):
                circuit = tq.QCircuit()
                for i in range(n_qubits):
                    circuit += tq.gates.Ry(angle=data[i], target=i)
                return circuit

            n_qubits = quantum_data.shape[1]
            circuits = [create_circuit(row, n_qubits) for row in quantum_data]

            # Calcular las expectativas para cada circuito usando Tequila
            expectation_values = []
            for circuit in circuits:
                # Simular el circuito con Tequila
                state = tq.simulate(circuit, backend="qulacs")  # Usar el backend adecuado
                expectation = np.real(state.inner(state))  # Expectativa del operador Z (por ejemplo)
                expectation_values.append(expectation)

            # Convertir las expectativas en una matriz numpy
            expectation_matrix = np.array(expectation_values)
            
            if expectation_matrix.ndim == 1:
                expectation_matrix = expectation_matrix.reshape(-1, 1)  # Convertir a 2D

            # Aplicar una técnica clásica de reducción de dimensiones (por ejemplo, PCA) a las expectativas
            n_samples, n_features = expectation_matrix.shape
            n_components = min(2, min(n_samples, n_features))

            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(expectation_matrix)
        
        elif(circuito_ansatz == 'Optimización de Parametros (QAOA)'):

            # Crear grafo
            def create_graph():
                G = nx.erdos_renyi_graph(12, 0.1)
                return G

            # Crear operador de costo
            def maxcut_operator(graph, num_qubits):
                pauli_list = []
                coeffs = []

                # Para cada arista del grafo, construir un término del Hamiltoniano
                for i, j in graph.edges():
                    if i >= num_qubits or j >= num_qubits:
                        raise IndexError(f"Índices de los nodos {i} y {j} están fuera del rango de qubits ({num_qubits})")

                    # Crear el pauli string
                    pauli_string = ['I'] * num_qubits

                    # Establecer las posiciones correspondientes de los nodos a 'Z'
                    pauli_string[i], pauli_string[j] = 'Z', 'Z'

                    # Añadir el término al operador
                    pauli_list.append(''.join(pauli_string))
                    coeffs.append(-1.0)

                return pauli_list, coeffs
            
            # Crear circuito QAOA
            def create_qaoa_circuit(num_qubits, params):
                layers = len(params) // 2
                circuit = tq.QCircuit()

                # Puertas Hadamard
                for qubit in range(num_qubits):
                    circuit += tq.gates.H(target=qubit)

                # Capas QAOA
                for i in range(layers):
                    gamma, beta = params[2 * i], params[2 * i + 1]
                    for j, k in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
                        circuit += tq.gates.CNOT(target=j, control=k)
                        circuit += tq.gates.Rz(angle=2 * gamma, target=k)
                        circuit += tq.gates.CNOT(target=j, control=k)
                    for qubit in range(num_qubits):
                        circuit += tq.gates.Rx(angle=2 * beta, target=qubit)

                return circuit
            
            # Ejecutar QAOA
            def run_qaoa(graph, p=1):
                num_qubits = graph.number_of_nodes()
                pauli_list, coeffs = maxcut_operator(graph, num_qubits)

                # Configurar backend
                backend = 'qulacs'  # Puedes cambiar el backend si lo deseas
                estimator = tq.simulate  # Usamos Tequila para simular

                # Inicialización de parámetros
                params = np.random.uniform(0, 2 * np.pi, size=2 * p)
                
                def pauli_string_to_dict(pauli_string):
                    pauli_dict = {}
                    for idx, p in enumerate(pauli_string):
                        if p != 'I':  # Solo consideramos los operadores no identidad
                            pauli_dict[idx] = p.lower()  # Convertir a minúscula para ser consistente
                    return pauli_dict

                # Función de costo
                def cost_function(params):
                    circuit = create_qaoa_circuit(num_qubits, params)
                    wavefunction = estimator(circuit, backend=backend)
                    
                    energy = 0
                    for pauli_string, coeff in zip(pauli_list, coeffs):
                        # Crear el operador de Pauli
                        pauli_dict = pauli_string_to_dict(pauli_string)
                        pauli_operator = tq.hamiltonian.PauliString(pauli_dict)
                        # Aplicar el PauliString al estado cuántico
                        result_wavefunction = wavefunction.apply_paulistring(pauli_operator)
                        # Calcular la expectativa del PauliString
                        energy += coeff * result_wavefunction.inner(other=wavefunction)
                        
                    return energy

                # Optimización clásica
                result = minimize(cost_function, params, method="COBYLA", options={"maxiter": 100})
                optimal_params = result.x
                optimal_energy = result.fun

                return optimal_params, optimal_energy
            
            graph = create_graph()
            print("Grafo:", graph.edges())

            # Resolver Max-Cut con QAOA
            p = 2
            params, energy = run_qaoa(graph, p=p)

            print("\nParámetros óptimos:", params)
            print("Energía mínima encontrada:", energy)

    # Agregar la vista inicial a la página
    page.add(login_view)

ft.app(target=main)
