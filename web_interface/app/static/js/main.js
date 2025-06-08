function updateClock() {
    const now = new Date();
    document.getElementById('current-time').textContent = 
        now.toISOString().replace('T', ' ').substr(0, 19);
}
setInterval(updateClock, 1000);
updateClock();
