// Main JavaScript file for School Recommendation System

document.addEventListener('DOMContentLoaded', function() {
    // Enable Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Enable Bootstrap popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Add animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.classList.add('shadow');
        });
        card.addEventListener('mouseleave', function() {
            this.classList.remove('shadow');
        });
    });

    // Initialize particle system
    initParticleSystem();
    
    // Initialize theme toggle
    initThemeToggle();
    
    // Initialize smooth scrolling
    initSmoothScrolling();
    
    // Initialize animations
    initAnimations();
    
    // Initialize mobile menu
    initMobileMenu();
});

// Particle System
function initParticleSystem() {
    const particlesContainer = document.getElementById('particles');
    if (!particlesContainer) return;
    
    const particleCount = 50;
    
    for (let i = 0; i < particleCount; i++) {
        createParticle(particlesContainer);
    }
}

function createParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    
    // Random position
    particle.style.left = Math.random() * 100 + '%';
    particle.style.top = Math.random() * 100 + '%';
    
    // Random animation delay
    particle.style.animationDelay = Math.random() * 6 + 's';
    
    // Random size
    const size = Math.random() * 3 + 1;
    particle.style.width = size + 'px';
    particle.style.height = size + 'px';
    
    // Add twinkle animation to some particles
    if (Math.random() > 0.7) {
        particle.style.animation += ', twinkle 3s ease-in-out infinite';
    }
    
    container.appendChild(particle);
    
    // Remove and recreate particle after animation
    setTimeout(() => {
        if (particle.parentNode) {
            particle.parentNode.removeChild(particle);
            createParticle(container);
        }
    }, (Math.random() * 10 + 10) * 1000);
}

// Theme Toggle
function initThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    if (!themeToggle) return;
    
    themeToggle.addEventListener('click', function() {
        document.body.classList.toggle('light-theme');
        
        const icon = themeToggle.querySelector('i');
        if (document.body.classList.contains('light-theme')) {
            icon.className = 'fas fa-sun';
        } else {
            icon.className = 'fas fa-moon';
        }
    });
}

// Smooth Scrolling
function initSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href === '#') return;
            
            const target = document.querySelector(href);
            if (target) {
                e.preventDefault();
                
                const offsetTop = target.offsetTop - 100; // Account for fixed navbar
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Animations
function initAnimations() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animatedElements = document.querySelectorAll('.stat-item, .process-step, .feature-item');
    animatedElements.forEach(el => {
        observer.observe(el);
    });
    
    // Counter animation for statistics
    animateCounters();
}

function animateCounters() {
    const counters = document.querySelectorAll('.stat-number');
    
    counters.forEach(counter => {
        const target = parseInt(counter.textContent.replace(/\D/g, ''));
        const suffix = counter.textContent.replace(/\d/g, '');
        let current = 0;
        const increment = target / 100;
        const duration = 2000; // 2 seconds
        const stepTime = duration / 100;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                counter.textContent = target + suffix;
                clearInterval(timer);
            } else {
                counter.textContent = Math.floor(current) + suffix;
            }
        }, stepTime);
    });
}

// Cosmic cursor effect
document.addEventListener('mousemove', function(e) {
    createCosmicTrail(e.clientX, e.clientY);
});

function createCosmicTrail(x, y) {
    const trail = document.createElement('div');
    trail.className = 'cosmic-trail';
    trail.style.cssText = `
        position: fixed;
        left: ${x}px;
        top: ${y}px;
        width: 4px;
        height: 4px;
        background: radial-gradient(circle, #8B5CF6, transparent);
        border-radius: 50%;
        pointer-events: none;
        z-index: 9999;
        animation: fadeOut 1s ease-out forwards;
    `;
    
    document.body.appendChild(trail);
    
    setTimeout(() => {
        if (trail.parentNode) {
            trail.parentNode.removeChild(trail);
        }
    }, 1000);
}

// Add CSS for cosmic trail animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        0% {
            opacity: 0.8;
            transform: scale(1);
        }
        100% {
            opacity: 0;
            transform: scale(0);
        }
    }
    
    .animate-in {
        opacity: 1;
        transform: translateY(0);
        transition: all 0.8s ease-out;
    }
    
    .stat-item,
    .process-step,
    .feature-item {
        opacity: 0;
        transform: translateY(30px);
        transition: all 0.8s ease-out;
    }
    
    .light-theme {
        --cosmic-purple: #6B46C1;
        --cosmic-blue: #0284C7;
        --cosmic-cyan: #DB2777;
        --deep-space: #F8FAFC;
        --star-white: #1E1E2E;
        --space-gray: #E5E7EB;
        --hero-gradient: linear-gradient(135deg, #F8FAFC 0%, #E5E7EB 50%, #D1D5DB 100%);
    }
    
    .light-theme .cosmic-background {
        background: var(--hero-gradient);
    }
    
    .light-theme body {
        color: var(--star-white);
    }
    
    .light-theme .sidebar {
        background: rgba(248, 250, 252, 0.95);
        border-right: 1px solid rgba(107, 70, 193, 0.2);
    }
    
    .light-theme .top-navbar {
        background: rgba(248, 250, 252, 0.95);
        border-bottom: 1px solid rgba(107, 70, 193, 0.2);
    }
`;
document.head.appendChild(style);

// Mobile Menu
function initMobileMenu() {
    const mobileMenuToggle = document.getElementById('mobileMenuToggle');
    const sidebar = document.querySelector('.sidebar');
    const mobileOverlay = document.getElementById('mobileOverlay');
    
    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener('click', function() {
            toggleMobileMenu();
        });
    }
    
    if (mobileOverlay) {
        mobileOverlay.addEventListener('click', function() {
            closeMobileMenu();
        });
    }
    
    // Close menu when clicking sidebar links on mobile
    const sidebarLinks = document.querySelectorAll('.sidebar .nav-link');
    sidebarLinks.forEach(link => {
        link.addEventListener('click', function() {
            if (window.innerWidth <= 768) {
                closeMobileMenu();
            }
        });
    });
}

function toggleMobileMenu() {
    const sidebar = document.querySelector('.sidebar');
    const mobileOverlay = document.getElementById('mobileOverlay');
    
    sidebar.classList.toggle('mobile-open');
    mobileOverlay.classList.toggle('active');
}

function closeMobileMenu() {
    const sidebar = document.querySelector('.sidebar');
    const mobileOverlay = document.getElementById('mobileOverlay');
    
    sidebar.classList.remove('mobile-open');
    mobileOverlay.classList.remove('active');
}

// Keyboard navigation support
document.addEventListener('keydown', function(e) {
    // ESC key to close mobile menu
    if (e.key === 'Escape') {
        closeMobileMenu();
    }
    
    // Space/Enter for buttons
    if ((e.key === ' ' || e.key === 'Enter') && e.target.classList.contains('theme-toggle')) {
        e.preventDefault();
        e.target.click();
    }
});

// Responsive particle count
function updateParticleCount() {
    const width = window.innerWidth;
    const particlesContainer = document.getElementById('particles');
    
    if (particlesContainer) {
        if (width < 768) {
            // Reduce particles on mobile
            const particles = particlesContainer.querySelectorAll('.particle');
            for (let i = particles.length - 1; i >= 20; i--) {
                particles[i].remove();
            }
        }
    }
}

window.addEventListener('resize', updateParticleCount);

// Performance optimization: Reduce animations on low-end devices
if (navigator.hardwareConcurrency && navigator.hardwareConcurrency < 4) {
    document.body.classList.add('reduced-motion');
    
    const style = document.createElement('style');
    style.textContent = `
        .reduced-motion * {
            animation-duration: 0.1s !important;
            transition-duration: 0.1s !important;
        }
        
        .reduced-motion .particle {
            display: none;
        }
    `;
    document.head.appendChild(style);
} 