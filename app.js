document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const uploadArea = document.getElementById('upload-area');
    const imageInput = document.getElementById('image-input');
    const imagePreview = document.getElementById('image-preview');
    const uploadContent = document.getElementById('upload-content');
    const contextToggle = document.getElementById('context-toggle');
    const accordion = document.querySelector('.accordion');
    const form = document.getElementById('advisory-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');
    
    // Result Elements
    const emptyState = document.getElementById('empty-state');
    const resultsContent = document.getElementById('results-content');
    
    let selectedFile = null;

    // --- Accordion Logic ---
    contextToggle.addEventListener('click', () => {
        accordion.classList.toggle('active');
    });

    // --- Drag and Drop Logic ---
    uploadArea.addEventListener('click', () => imageInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    imageInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFile(this.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        
        selectedFile = file;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            uploadContent.style.display = 'none';
            uploadArea.style.padding = '0';
            uploadArea.style.border = 'none';
            submitBtn.disabled = false;
        }
        reader.readAsDataURL(file);
    }

    // --- Form Submission Logic ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        // UI Loading state
        submitBtn.disabled = true;
        btnText.style.display = 'none';
        spinner.style.display = 'block';
        
        emptyState.style.display = 'none';
        resultsContent.style.display = 'none';

        // Gather Metadata
        const metadata = {
            district_std: document.getElementById('district').value,
            season: document.getElementById('season').value,
            crop_year: parseInt(document.getElementById('crop_year').value),
            area_ha: parseFloat(document.getElementById('area_ha').value),
            crop_type: "maize",
            growth_stage: document.getElementById('growth_stage').value,
            language: document.getElementById('language').value,
            ...weatherData // from index.html script tag
        };

        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('metadata', JSON.stringify(metadata));

        try {
            const response = await fetch('/orchestrate', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (!response.ok) {
                const errorMsg = data.detail || `Server error: ${response.status}`;
                throw new Error(errorMsg);
            }

            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert('Analysis Error: ' + error.message);
            emptyState.style.display = 'flex';
        } finally {
            // Restore UI
            submitBtn.disabled = false;
            btnText.style.display = 'block';
            spinner.style.display = 'none';
        }
    });

    function displayResults(data) {
        // Show results container
        resultsContent.style.display = 'block';
        
        // 1. Visual Diagnosis
        const diag = data.visual_diagnosis || {};
        const diagStr = diag.diagnosis || "Unknown";
        document.getElementById('res-diagnosis').innerText = diagStr;
        
        const confText = document.getElementById('res-confidence-text');
        const confBar = document.getElementById('res-confidence-bar');
        const confPercentage = Math.round((diag.confidence || 0) * 100);
        confText.innerText = `${confPercentage}%`;
        
        // Animate progress bar
        confBar.style.width = '0%';
        setTimeout(() => {
            confBar.style.width = `${confPercentage}%`;
        }, 100);

        const badge = document.getElementById('severity-badge');
        const severity = (diag.severity || "low").toLowerCase();
        badge.innerText = severity.charAt(0).toUpperCase() + severity.slice(1) + " Severity";
        badge.className = `badge ${severity}`;

        const img = document.getElementById('result-image');
        if (diag.annotated_image_base64) {
             img.src = `data:image/jpeg;base64,${diag.annotated_image_base64}`;
        } else {
             img.src = imagePreview.src;
        }

        // 2. Yield Projection
        const env = data.environmental_context || {};
        const yieldVal = env.expected_yield_baseline || "N/A";
        document.getElementById('res-yield').innerText = yieldVal.replace(' t/ha', '');
        document.getElementById('res-district').innerText = env.district || "Unknown";
        document.getElementById('res-season').innerText = env.season || "Unknown";

        // 3. Advisory text
        const advisoryDiv = document.getElementById('res-advisory');
        const expert = data.expert_advisory || {};
        let advisoryText = expert.advisory || JSON.stringify(expert, null, 2);
        
        // Very simple markdown to html conversion for bold/lists if LLM returns markdown
        advisoryText = advisoryText
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n- /g, '<br>• ');

        advisoryDiv.innerHTML = `<div class="advisory-text">${advisoryText}</div>`;
        
        // Scroll to results on mobile
        if(window.innerWidth < 900) {
            advisoryDiv.scrollIntoView({ behavior: 'smooth' });
        }
    }
});
