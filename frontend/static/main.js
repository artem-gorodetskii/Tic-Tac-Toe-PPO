const boardEl = document.getElementById('board');
const modal = document.getElementById('modal');
const modalContent = document.getElementById('modal-content');
const slidingWindowSize = 1000;
const ctx = document.getElementById('train-chart').getContext('2d');

let CONFIG;
let board = [];
let agentStarts = false;
let gameOver = false;
let winningLine = [];
let isTraining = false;
let eventSource = null;
let maxWinRate = 0;
let selectedOpponent = document.querySelector('.opponent-btn.active')?.dataset.opponent || 'random';

// Load config from the backend and start the game.
fetch('http://localhost:8000/env_config')
  .then(res => res.json())
  .then(config => {
    CONFIG = config;
    restart();
  })
  .catch(err => {
    console.error('Failed to load config:', err);
    modalContent.textContent = 'Failed to load game config';
    modal.classList.add('show');
  });

// Render the game board.
function render() {
  const size = CONFIG.board_size;
  boardEl.innerHTML = '';

  for (let i = 0; i < size; i++) {
    const tr = document.createElement('tr');
    for (let j = 0; j < size; j++) {
      const index = i * size + j;
      const td = document.createElement('td');

      if (board[index] === CONFIG.opponent_player) {
        td.textContent = 'X';
      } else if (board[index] === CONFIG.agent_player) {
        td.textContent = 'O';
      } else {
        td.textContent = '';
      }

      td.classList.toggle('played', board[index] !== CONFIG.free_cell_val);
      td.classList.toggle('win-cell', winningLine.includes(index));

      if (!isTraining && !gameOver && board[index] === CONFIG.free_cell_val) {
        td.onclick = () => handleMove(index);
        td.style.cursor = 'pointer';
      } else {
        td.style.cursor = 'default';
      }
      tr.appendChild(td);
    }
    boardEl.appendChild(tr);
  }
}

// Handle the agent move.
function handleMove(index) {
  if (isTraining || gameOver || board[index] !== CONFIG.free_cell_val) return;
  board[index] = CONFIG.opponent_player;
  render();

  fetch('http://localhost:8000/agent-play', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ board })
  })
    .then(res => res.json())
    .then(data => {
      if (data.move !== -1) {
        board[data.move] = CONFIG.agent_player;
      }

      if (data.gameOver) {
        showResult(data.winner, data.line);
      } else {
        render();
      }
    })
    .catch(console.error);
}

// Show result: win or draw.
function showResult(winner, line = []) {
  winningLine = line;
  render();

  let msg = '';
  if (winner === CONFIG.opponent_player) msg = "You won!";
  else if (winner === CONFIG.agent_player) msg = "Agent won!";
  else msg = "It's a draw!";

  modalContent.textContent = msg;
  modal.classList.add('show');
  gameOver = true;
}

// Clear the board.
function clearBoard() {
  board = Array(CONFIG.board_size * CONFIG.board_size).fill(CONFIG.free_cell_val);
  gameOver = false;
  winningLine = [];
  render();
  modal.classList.remove('show');
}

// Restart the game.
function restart() {
  clearBoard()
  setStatus(agentStarts ? "Agent starts" : "You start");
  render();

  if (agentStarts) {
    setTimeout(() => {
      fetch('http://localhost:8000/agent-play', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board })
      })
        .then(res => res.json())
        .then(data => {
          if (data.move !== -1) {
            board[data.move] = CONFIG.agent_player;
          }
          if (data.gameOver) {
            showResult(data.winner, data.line);
          } else {
            render();
          }
        })
        .catch(console.error);
    }, 200);
  } else {
    render();
  }
  agentStarts = !agentStarts;
}

// Set the board status.
function setStatus(message) {
  document.getElementById('status').textContent = message;
}

// Attache a click handler to the modal window.
modal.addEventListener('click', (e) => {
  if (e.target === modal) modal.classList.remove('show');
});

// Show notification.
function showModalMessage(message) {
  modalContent.textContent = message;
  modal.classList.add('show');
}

// Training chart.
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    datasets: [
    {
      label: 'Agent Wins (last 100)',
      data: [],
      borderColor: getComputedStyle(document.documentElement)
        .getPropertyValue('--border-color').trim(),
      backgroundColor: 'rgba(76, 175, 80, 0.1)',
      fill: true,
      tension: 0.2,
      pointRadius: 0,
      pointHoverRadius: 5,
      order: 1,
    },
    {
      label: 'Best Win Rate',
      data: [],
      borderColor: 'rgba(255, 0, 0, 0.5)',
      borderDash: [5, 5],
      pointRadius: 0,
      borderWidth: 2.5,
      fill: false,
      order: 10,
    }
  ]
  },
  options: {
    responsive: true,
    animation: false,
    scales: {
      x: {
        type: 'linear',
        title: { display: true, text: 'Step' },
        ticks: {
          autoSkip: false,
          callback: value => Number.isInteger(value) ? value : '',
          maxTicksLimit: 10,
        },
        grid: { display: false }
      },
      y: {
        title: { display: true, text: 'Wins (out of 100)' },
        min: 0,
        max: 100,
        ticks: { stepSize: 10 },
        grid: {
          display: true,
          color: 'rgba(255, 255, 255, 0.1)',
          lineWidth: 1.5,
        }
      }
    },
    plugins: {
      legend: {
        labels: {
          usePointStyle: true,
          pointStyle: 'line',
          filter: (legendItem, data) => {
            return legendItem.text == 'Best Win Rate';
          },
        }
      }
    }
  }
});

// Control the behavior of the training button.
document.getElementById('train-btn').onclick = async () => {
  const trainBtn = document.getElementById('train-btn');

  if (!isTraining) {
    isTraining = true;
    trainBtn.style.backgroundColor = '#888';
    trainBtn.textContent = '⏸ Stop Training';
    startTraining();
  } else {
    stopTraining();
  }
};

// Start agent training.
async function startTraining() {
  document.getElementById('next-round-btn').disabled = true;
  setStatus("Blocked during training");
  clearBoard()

  chart.data.datasets[0].data = [];
  chart.data.datasets[1].data = [];
  maxWinRate = 0;
  delete chart.options.scales.x.min;
  delete chart.options.scales.x.max;
  chart.update();

  await fetch('http://localhost:8000/train');
  eventSource = new EventSource('http://localhost:8000/train/stream');

  eventSource.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    chart.data.datasets[0].data.push({ x: data.step, y: data.rolling_win_rate });

    if (data.rolling_win_rate > maxWinRate) {
      maxWinRate = data.rolling_win_rate;
    }

    const startX = chart.data.datasets[0].data[0]?.x ?? data.step;
    chart.data.datasets[1].data = [
      { x: startX, y: maxWinRate },
      { x: data.step, y: maxWinRate }
    ];
  
    if (chart.data.datasets[0].data.length > slidingWindowSize) {
      chart.data.datasets[0].data.shift();
    }
    chart.options.scales.x.min = data.step - slidingWindowSize + 1;
    chart.options.scales.x.max = data.step;
    chart.update();
  });

  eventSource.addEventListener('stopped', () => {
    stopTraining();
  });

  eventSource.onerror = () => {
    stopTraining();
  };
}

// Stop agent trainig.
async function stopTraining() {
  if (eventSource) {
    eventSource.onerror = null;
    eventSource.close();
    eventSource = null;
  }

  const trainBtn = document.getElementById('train-btn');
  trainBtn.textContent = 'Stopping...';
  await fetch('http://localhost:8000/stop-training');

  isTraining = false;
  trainBtn.disabled = false;
  trainBtn.style.backgroundColor = '';
  trainBtn.textContent = '▶ Run Training';
  document.getElementById('next-round-btn').disabled = false;

  agentStarts = false;
  restart();
}

// Save checkpoint.
document.getElementById('save-btn').onclick = async () => {
  try {
    const res = await fetch('http://localhost:8000/save_checkpoint', {
      method: 'POST',
    });
    const data = await res.json();
    showModalMessage(data.status);
  } catch (err) {
    showModalMessage('Error saving checkpoint');
    console.error(err);
  }
};

// Load pretrained agent.
document.getElementById('load-btn').onclick = async () => {
  try {
    const res = await fetch('http://localhost:8000/load_checkpoint', {
      method: 'POST',
    });
    const data = await res.json();
    agentStarts = false;
    restart()
    showModalMessage(data.status);
  } catch (err) {
    showModalMessage('Error loading checkpoint');
    console.error(err);
  }
};

// Select an agent opponent for training.
document.querySelectorAll('.opponent-btn').forEach(btn => {

  btn.addEventListener('click', async (e) => {
    const opponent = e.currentTarget.dataset.opponent;
  
    if (selectedOpponent === opponent) {
      return;
    }

    if (isTraining) {
      modalContent.textContent = 'Stop training before changing opponent';
      modal.classList.add('show');
      return;
    }

    try {
      const res = await fetch('http://localhost:8000/set_opponent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ opponent })
      });
  
      const data = await res.json();

      if (data.status === 'ok') {
        selectedOpponent = opponent;
        document.querySelectorAll('.opponent-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
      } else {
        modalContent.textContent = data.error || 'Failed to change opponent';
        modal.classList.add('show');
      }
    } catch (err) {
      modalContent.textContent = 'Failed to connect to backend';
      modal.classList.add('show');
    }
  });
});
