document.addEventListener('mousemove', (e) => {
    const fleur = document.createElement('div');
    fleur.classList.add('fleur');

    // Fleur emoji (tu peux les remplacer par des images si tu veux)
    const fleurs = ['🌸', '🌼', '🌺', '💮', '🌷'];
    fleur.textContent = fleurs[Math.floor(Math.random() * fleurs.length)];

    // Positionnement
    fleur.style.left = `${e.pageX}px`;
    fleur.style.top = `${e.pageY}px`;

    // Ajout à la page
    document.body.appendChild(fleur);

    // Suppression après l’animation
    setTimeout(() => {
        fleur.remove();
    }, 1000);
});
