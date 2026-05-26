# Lale

[![Tests](https://github.com/IBM/lale/actions/workflows/build.yml/badge.svg)](https://github.com/IBM/lale/actions/workflows/build.yml)[![Documentation Status](https://readthedocs.org/projects/lale/badge/?version=latest)](https://lale.readthedocs.io/en/latest/?badge=latest)
[![PyPI version shields.io](https://img.shields.io/pypi/v/lale?color=success)](https://pypi.python.org/pypi/lale/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5863/badge)](https://bestpractices.coreinfrastructure.org/projects/5863)
<br />
<img src="https://github.com/IBM/lale/raw/master/docs/img/lale_logo.jpg" alt="logo" width="55px"/>

<div dir="rtl" align="right">

<span dir="ltr">Lale</span> یک کتابخانه <span dir="ltr">Python</span> برای علم داده نیمه‌ خودکار است. این کتابخانه، انتخاب خودکار الگوریتم‌ها و تنظیم ابرپارامترهای پایپ‌لاین‌های سازگار با <span dir="ltr">scikit-learn</span> را، به شکلی ایمن از نظر نوع داده ساده می‌کند. اگر دانشمند داده هستید و می‌خواهید یادگیری ماشین خودکار را آزمایش و بررسی کنید، این کتابخانه می‌تواند برای شما مفید باشد.

<span dir="ltr">Lale</span> در سه زمینه، قابلیت هایی فراتر از <span dir="ltr">scikit-learn</span>  ایجاد می‌کند: خودکارسازی، بررسی درستی، و تعامل‌پذیری. برای خودکارسازی، <span dir="ltr">Lale</span> یک رابط سطح‌بالای یکپارچه برای ابزارهای موجود جست‌وجوی پایپ‌لاین مانند <span dir="ltr">Hyperopt</span>، <span dir="ltr">GridSearchCV</span> و <span dir="ltr">SMAC</span> فراهم می‌کند. برای بررسی درستی، <span dir="ltr">Lale</span> از <span dir="ltr">JSON Schema</span> استفاده می‌کند تا ناسازگاری‌های مربوط به ابرپارامترها و نوع آن‌ها، یا ناسازگاری‌های مربوط به داده‌ها و عملگرها را شناسایی کند. و برای تعامل‌پذیری، <span dir="ltr">Lale</span> مجموعه‌ای رو به رشد از تبدیل‌گرها و تخمین‌گرهای کتابخانه‌های محبوبی مانند <span dir="ltr">scikit-learn</span>، <span dir="ltr">XGBoost</span>، <span dir="ltr">PyTorch</span> و سایر کتابخانه‌های مشابه دیگر را در اختیار دارد. <span dir="ltr">Lale</span> مانند هر بسته <span dir="ltr">Python</span> دیگری قابل نصب است و همچنین با استفاده از ابزارهای رایج <span dir="ltr">Python</span> مانند <span dir="ltr">Jupyter notebooks</span> قابل ویرایش است.

- [راهنمای مقدماتی](https://github.com/IBM/lale/blob/master/examples/docs_guide_for_sklearn_users.ipynb) برای کاربران <span dir="ltr">scikit-learn</span>
- [دستورالعمل نصب](https://github.com/IBM/lale/blob/master/docs/installation.rst)
- مرور فنی: [اسلایدها](https://github.com/IBM/lale/blob/master/talks/2019-1105-lale.pdf)، [نوت‌بوک](https://github.com/IBM/lale/blob/master/examples/talk_2019-1105-lale.ipynb)، و [ویدئو](https://www.youtube.com/watch?v=R51ZDJ64X18&list=PLGVZCDnMOq0pwoOqsaA87cAoNM4MWr51M&index=35&t=0s)
- <span dir="ltr">IBM [AutoAI SDK](http://wml-api-pyclient-v4.mybluemix.net/#autoai-beta-ibm-cloud-only)</span> از <span dir="ltr">Lale</span> استفاده می‌کند؛ نسخه دمو [نوت‌بوک](https://dataplatform.cloud.ibm.com/exchange/public/entry/view/8bddf7f7e5d004a009c643750b16d0c0)  را ببینید
- راهنمای اضافه کردن [عملگرهای جدید](https://github.com/IBM/lale/blob/master/examples/docs_new_operators.ipynb)
- راهنمای [مشارکت](https://github.com/IBM/lale/blob/master/CONTRIBUTING.md) در <span dir="ltr">Lale</span>
- [پرسش‌های متداول](https://github.com/IBM/lale/blob/master/docs/faq.rst)
- [مقالات](https://github.com/IBM/lale/blob/master/docs/papers.rst)
- [مستندات <span dir="ltr">API</span>](https://lale.readthedocs.io/en/latest/) پایتون

نام <span dir="ltr">Lale</span>، که به صورت <em>laleh</em> تلفظ می‌شود، از واژه فارسی «لاله» گرفته شده است. مانند سایر کتابخانه‌های محبوب یادگیری ماشین، از جمله <span dir="ltr">scikit-learn</span>، ‏<span dir="ltr">Lale</span> نیز فقط یک کتابخانه <span dir="ltr">Python</span> است، نه یک زبان برنامه‌نویسی مستقل جدید. کاربران لازم نیست ابزارهای جدیدی نصب کنند یا <span dir="ltr">syntax</span> جدیدی یاد بگیرند.

<span dir="ltr">Lale</span> تحت شرایط مجوز <span dir="ltr">Apache 2.0</span> منتشر شده است؛ برای اطلاعات بیشتر فایل [LICENSE.txt](https://github.com/IBM/lale/blob/master/LICENSE.txt) را ببینید. این پروژه در حال حاضر در وضعیت **نسخه آلفا** است و هیچ‌گونه ضمانتی ندارد.

</div>
