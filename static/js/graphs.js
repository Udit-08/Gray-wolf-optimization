document.addEventListener("DOMContentLoaded", () => {
    
    // UI Elements
    const DOM = {
        refreshBtn: document.getElementById('refreshBtn'),
        statusAlert: document.getElementById('statusAlert'),
        dashboard: document.getElementById('dashboard'),
        
        val_lr_acc: document.getElementById('val_lr_acc'),
        val_svm_acc: document.getElementById('val_svm_acc'),
        val_lr_f1: document.getElementById('val_lr_f1'),
        val_svm_f1: document.getElementById('val_svm_f1'),
        
        lrMatrixContainer: document.getElementById('lrMatrixContainer'),
        svmMatrixContainer: document.getElementById('svmMatrixContainer')
    };

    // Charts instances
    let compChart = null;
    let convChart = null;

    // Helper: Show Alert
    const showAlert = (msg, isError = false) => {
        DOM.statusAlert.textContent = msg;
        DOM.statusAlert.className = `p-4 rounded-lg text-sm font-bold flex items-center gap-2 mb-6 ${isError ? 'bg-red-500/10 text-red-400 border border-red-500/30' : 'bg-primary/10 text-primary border border-primary/30'}`;
        DOM.statusAlert.classList.remove('hidden');
    };

    // Helper: Render Confusion Matrix as HTML Grid
    const renderMatrix = (container, matrix, mode) => {
        container.innerHTML = '';
        
        const size = matrix.length;
        const labels = mode === 'binary' ? ['Negative (0)', 'Positive (1)'] : ['Class 0', 'Class 1', 'Class 2'];
        
        // Find max value for color scaling
        let maxVal = 0;
        for(let i=0; i<size; i++) {
            for(let j=0; j<size; j++) {
                if(matrix[i][j] > maxVal) maxVal = matrix[i][j];
            }
        }

        // Create Grid Wrapper
        const wrapper = document.createElement('div');
        wrapper.className = `grid gap-1`;
        wrapper.style.gridTemplateColumns = `auto repeat(${size}, minmax(0, 1fr))`;
        
        // Top Left Empty
        wrapper.appendChild(document.createElement('div'));
        
        // Col Headers (Predicted)
        for(let i=0; i<size; i++) {
            const h = document.createElement('div');
            h.className = "text-[10px] uppercase text-on-surface-variant font-bold text-center self-end pb-2";
            h.innerHTML = `Pred<br>${labels[i]}`;
            wrapper.appendChild(h);
        }

        // Rows
        for(let i=0; i<size; i++) {
            // Row Header (True)
            const rh = document.createElement('div');
            rh.className = "text-[10px] uppercase text-on-surface-variant font-bold text-right pr-4 self-center";
            rh.innerHTML = `True<br>${labels[i]}`;
            wrapper.appendChild(rh);
            
            // Cells
            for(let j=0; j<size; j++) {
                const cell = document.createElement('div');
                const val = matrix[i][j];
                const intensity = maxVal === 0 ? 0 : Math.max(0.1, val / maxVal);
                
                // Color logic: Diagonal (correct) gets green/blue, off-diagonal gets red/orange
                let colorBase = i === j ? `rgba(46, 204, 113, ${intensity})` : `rgba(255, 110, 132, ${intensity})`;
                
                cell.className = "heatmap-cell aspect-square text-lg shadow-inner";
                cell.style.backgroundColor = colorBase;
                cell.textContent = val;
                
                wrapper.appendChild(cell);
            }
        }
        
        container.appendChild(wrapper);
    };

    // Main Fetch Function
    const fetchAndRender = async () => {
        try {
            DOM.refreshBtn.disabled = true;
            DOM.statusAlert.classList.add('hidden');
            
            const res = await fetch('/api/metrics');
            const data = await res.json();
            
            if(!res.ok) throw new Error(data.error || "Failed to fetch metrics.");

            // 1. Update Values
            DOM.val_lr_acc.textContent = (data.lr.accuracy * 100).toFixed(1) + "%";
            DOM.val_svm_acc.textContent = (data.svm.accuracy * 100).toFixed(1) + "%";
            DOM.val_lr_f1.textContent = data.lr.f1.toFixed(3);
            DOM.val_svm_f1.textContent = data.svm.f1.toFixed(3);

            // 2. Render Matrices
            renderMatrix(DOM.lrMatrixContainer, data.lr.confusion_matrix, data.mode);
            renderMatrix(DOM.svmMatrixContainer, data.svm.confusion_matrix, data.mode);

            // 3. Comparison Chart
            const compCtx = document.getElementById('comparisonChart').getContext('2d');
            if(compChart) compChart.destroy();
            compChart = new Chart(compCtx, {
                type: 'bar',
                data: {
                    labels: ['Accuracy', 'F1-Score (Weighted)'],
                    datasets: [
                        {
                            label: 'Logistic Regression (GWO)',
                            data: [data.lr.accuracy, data.lr.f1],
                            backgroundColor: '#df8eff',
                            borderRadius: 4
                        },
                        {
                            label: 'Support Vector Machine',
                            data: [data.svm.accuracy, data.svm.f1],
                            backgroundColor: '#00e3fd',
                            borderRadius: 4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { 
                            beginAtZero: true, max: 1.0, 
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#aaa8c2' }
                        },
                        x: { 
                            grid: { display: false },
                            ticks: { color: '#aaa8c2', font: { weight: 'bold' } }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#e5e3fe', usePointStyle: true, boxWidth: 8 } }
                    }
                }
            });

            // 4. Convergence Chart
            const convCtx = document.getElementById('convergenceChart').getContext('2d');
            if(convChart) convChart.destroy();
            
            const gradient = convCtx.createLinearGradient(0, 0, 0, 400);
            gradient.addColorStop(0, 'rgba(223, 142, 255, 0.4)');
            gradient.addColorStop(1, 'rgba(223, 142, 255, 0)');

            const history = data.fitness_history || [];
            const labels = history.map((_, i) => `Iter ${i+1}`);

            convChart = new Chart(convCtx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Best Fitness (Alpha Wolf)',
                        data: history,
                        borderColor: '#df8eff',
                        backgroundColor: gradient,
                        borderWidth: 2,
                        pointBackgroundColor: '#fff',
                        pointBorderColor: '#df8eff',
                        pointRadius: 3,
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { 
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#aaa8c2' }
                        },
                        x: { 
                            grid: { display: false },
                            ticks: { color: '#aaa8c2', maxTicksLimit: 10 }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });

            // Show Dashboard
            DOM.dashboard.classList.remove('opacity-0', 'pointer-events-none');
            showAlert("Metrics successfully loaded and visualized.", false);

        } catch (err) {
            showAlert(`Error: ${err.message}. Have you initialized the training on the Diagnostics page?`, true);
        } finally {
            DOM.refreshBtn.disabled = false;
        }
    };

    // Initial Fetch
    fetchAndRender();

    // Bind Refresh
    DOM.refreshBtn.addEventListener('click', fetchAndRender);
});
