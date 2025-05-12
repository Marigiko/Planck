"use client"

import { useEffect, useRef } from "react"
import Chart from "chart.js/auto"

export function BenchmarkChartAlt() {
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

    // Datos para el gráfico (basados en la segunda imagen compartida)
    const labels = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    const qiskitData = [11.2, 10.8, 10.0, 10.3, 10.8, 10.9, 11.0]
    const ibmqData = [11.1, 11.5, 11.8, 12.5, 13.1, 11.5, 10.9]
    const openqasmData = [10.9, 10.8, 10.7, 11.2, 11.6, 10.8, 10.5]

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
            min: 10.0,
            max: 13.5,
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
