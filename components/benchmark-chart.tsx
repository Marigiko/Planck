"use client"

import { useEffect, useRef } from "react"
import Chart from "chart.js/auto"

export function BenchmarkChart() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)

  useEffect(() => {
    if (!chartRef.current) return

    // Destruir el gráfico anterior si existe
    if (chartInstance.current) {
      chartInstance.current.destroy()
    }

    const ctx = chartRef.current.getContext("2d")
    if (!ctx) return

    // Datos para el gráfico (basados en la imagen compartida)
    const labels = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    const qiskitData = [11.7, 10.9, 10.9, 10.5, 10.5, 10.5, 11.6, 10.3, 11.4]
    const ibmqData = [12.6, 10.7, 10.9, 12.8, 10.9, 11.0, 10.5, 11.1, 10.0]
    const openqasmData = [10.7, 11.5, 10.4, 11.5, 11.3, 10.5, 10.3, 12.1, 9.8]

    chartInstance.current = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "qiskit",
            data: qiskitData,
            borderColor: "#3b82f6", // chart.blue
            backgroundColor: "rgba(59, 130, 246, 0.1)",
            tension: 0.1,
            pointRadius: 4,
          },
          {
            label: "ibmq",
            data: ibmqData,
            borderColor: "#f97316", // chart.orange
            backgroundColor: "rgba(249, 115, 22, 0.1)",
            tension: 0.1,
            pointRadius: 4,
          },
          {
            label: "openqasm",
            data: openqasmData,
            borderColor: "#10b981", // chart.green
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            tension: 0.1,
            pointRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: "Benchmark QAOA - Tiempo de Ejecución",
            font: {
              size: 16,
              weight: "bold",
            },
          },
          legend: {
            position: "top",
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: "Número de Qubits",
            },
            grid: {
              display: true,
            },
          },
          y: {
            title: {
              display: true,
              text: "Tiempo de Ejecución (s)",
            },
            grid: {
              display: true,
            },
            min: 9.5,
            max: 13.0,
          },
        },
      },
    })

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy()
      }
    }
  }, [])

  return <canvas ref={chartRef} />
}
