// script.js
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const results = document.getElementById('results');
const historyList = document.getElementById('historyList');
const processBtn = document.getElementById('processBtn');
const fileNameSpan = document.getElementById('fileName');
const toastContainer = document.getElementById('toastContainer');

const maxHistory = 5;
let selectedFile = null;

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) selectFile(e.dataTransfer.files[0]);
});

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) selectFile(fileInput.files[0]);
});

function selectFile(file) {
  selectedFile = file;
  fileNameSpan.textContent = file.name;
}

processBtn.addEventListener('click', () => {
  if (!selectedFile) {
    showToast("Please select a CSV file!");
    return;
  }
  if (selectedFile.type !== "text/csv") {
    showToast("Invalid file type. Only CSV allowed.");
    resetSelection();
    return;
  }

  // Ignore file content, display preset result
   sendQuery(selectedFile);
  resetSelection();
});
function sendQuery(file) {
    const reader = new FileReader();
    reader.onload = function(event) {
        let dataLoad = {}
        dataLoad["rawCSV"] = event.target.result;
        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', },
            body: JSON.stringify(dataLoad),
            }
        )
        .then(response => response.json())
        .then(data => {
          displayResults(data["P(G4)"],data["MYC"],data["TP53"],data["SNCAIP"]);
        })
        .catch((error) => {
            console.error('Error:', error);
            }
        );
    };
    reader.readAsText(selectedFile);
}

function displayResults(result, v1, v2,v3) {
  results.innerHTML = `
    <div class="row align-items-center animate__animated animate__fadeIn">
      <div class="col-md-4 text-center">
        <h2 class="display-8 fw-bold text-secondary">P(G4):</h2>
        <p class="display-2 fw-bold text-primary">${result}</p>
      </div>
      <div class="col-md-8">
        <table class="table table-bordered w-100">
          <tbody>
            <tr class="table-primary"><td class="fw-semibold">MYC</td><td class="fw-bold fs-5">${v1}</td></tr>
            <tr class="table-success"><td class="fw-semibold">TP53</td><td class="fw-bold fs-5">${v2}</td></tr>
            <tr class="table-warning"><td class="fw-semibold">SNCAIP</td><td class="fw-bold fs-5">${v3}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  `;

  renderHistoryRow(result, v1, v2, v3);
}

var isFirstTime = true;
function renderHistoryRow(resultValue, val1, val2, val3) {
  const li = document.createElement('li');
  li.className = 'list-group-item d-flex justify-content-between align-items-center animate__animated animate__fadeIn';
  li.innerHTML = `
    <span class="fw-bold fs-5 text-primary">P(G4): ${resultValue}</span>
    <span class="badge bg-primary rounded-pill fw-bold fs-6">MYC: ${val1}</span>
    <span class="badge bg-success rounded-pill fw-bold fs-6">TP53: ${val2}</span>
    <span class="badge bg-warning text-dark rounded-pill fw-bold fs-6">SNCAIP: ${val3}</span>
  `;
  if(isFirstTime){
    historyList.innerHTML = ''; // Only one row in history
    isFirstTime = false;
    }
  historyList.appendChild(li);
}

function resetSelection() {
  selectedFile = null;
  fileInput.value = '';
  fileNameSpan.textContent = "Drag & Drop your CSV here or click to browse";
}

function showToast(message) {
  const toastEl = document.createElement('div');
  toastEl.className = 'toast show align-items-center text-bg-danger border-0 custom-toast';
  toastEl.setAttribute('role', 'alert');
  toastEl.setAttribute('aria-live', 'assertive');
  toastEl.setAttribute('aria-atomic', 'true');

  toastEl.innerHTML = `
    <div class="d-flex flex-column">
      <div class="toast-body">${message}</div>
      <div class="progress-bar-bg"><div class="progress-bar-fill"></div></div>
    </div>
  `;

  toastContainer.appendChild(toastEl);

  setTimeout(() => {
    toastEl.classList.remove('show');
    toastEl.remove();
  }, 3000);
}
