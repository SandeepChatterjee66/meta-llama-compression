document.addEventListener('DOMContentLoaded', function() {
    var elems = document.querySelectorAll('.carousel');
    var instances = M.Carousel.init(elems, {
        fullWidth: true,
        indicators: true
    });

    var carousel = document.querySelector('.carousel')
    if(carousel){
        carousel.style.height = '280px';
    }
    
    submit = document.getElementById('submit-button');
    
    if(submit){
        submit.addEventListener('click', function() {
            // Simulate progress
            window.location.href = '/progress';
        });
    }
});
