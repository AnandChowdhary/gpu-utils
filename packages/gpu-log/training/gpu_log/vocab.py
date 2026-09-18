"""Vocabulary pools for the synthetic log generator."""

from __future__ import annotations

HOSTS = [
    "web-01", "web-02", "api-prod-3", "db01", "cache-7", "ip-10-0-1-23", "node-42", "worker-b",
    "k8s-master", "localhost", "srv12.example.com", "prod-app-1.internal", "myhost", "combo",
    "LabSZ", "ubuntu", "macbook-pro", "raspberrypi", "build-agent-7", "ip-172-31-4-88",
    "gke-cluster-default-pool-1a2b3c4d-xyz1", "vm-eu-west-1b-004", "edge-gw-2", "mail",
]
PROCS = [
    "sshd", "systemd", "kernel", "cron", "CRON", "nginx", "postfix/smtpd", "dockerd", "kubelet",
    "containerd", "sudo", "su", "dhclient", "NetworkManager", "rsyslogd", "haproxy", "redis-server",
    "postgres", "mysqld", "php-fpm", "gunicorn", "uwsgi", "node", "java", "python3", "app",
    "sshd(pam_unix)", "systemd-logind", "snapd", "chronyd", "audit", "puppet-agent", "collectd",
    "consul", "vault", "envoy", "traefik", "caddy", "celery", "sidekiq", "puma", "unicorn",
]
JAVA_LOGGERS = [
    "com.example.api.UserController", "org.springframework.boot.SpringApplication",
    "org.apache.kafka.clients.NetworkClient", "com.zaxxer.hikari.HikariDataSource",
    "org.hibernate.SQL", "io.netty.channel.DefaultChannelPipeline", "o.s.web.servlet.DispatcherServlet",
    "c.e.d.Application", "org.apache.catalina.core.StandardService", "com.acme.billing.InvoiceService",
    "org.elasticsearch.cluster.service.MasterService", "org.apache.zookeeper.server.NIOServerCnxn",
    "dfs.DataNode$PacketResponder", "org.apache.hadoop.hdfs.server.namenode.FSNamesystem",
    "kafka.server.KafkaServer", "com.google.common.cache.LocalCache", "ch.qos.logback.classic.LoggerContext",
    "io.grpc.netty.NettyServerHandler", "com.example.service.PaymentGateway", "org.quartz.core.QuartzScheduler",
    "akka.actor.ActorSystemImpl", "reactor.netty.http.server.HttpServer", "com.mongodb.driver.cluster",
    "UserController", "Main", "Scheduler", "HttpClient", "DbPool", "AuthFilter",
]
PY_LOGGERS = [
    "root", "__main__", "app.api", "django.request", "django.server", "urllib3.connectionpool",
    "werkzeug", "uvicorn.error", "uvicorn.access", "gunicorn.error", "celery.worker", "sqlalchemy.engine",
    "botocore.credentials", "requests.packages.urllib3", "myapp.services.payments", "airflow.task",
    "scrapy.core.engine", "asyncio", "aiohttp.access", "flask.app", "app", "worker", "pipeline.ingest",
    "torch.distributed", "transformers.modeling_utils", "kombu.connection", "paramiko.transport",
]
GO_PKGS = [
    "main", "server", "pkg/api", "internal/db", "github.com/acme/app/handler", "cmd/agent",
    "pkg/scheduler", "controller", "reconciler", "k8s.io/client-go/tools/cache", "storage/blob",
    "grpc", "http", "worker", "queue", "auth", "cache", "metrics", "proxy", "router",
]
RUST_TARGETS = [
    "app", "app::server", "hyper::proto::h1::io", "tokio::runtime", "sqlx::query", "actix_web::middleware::logger",
    "reqwest::connect", "tower_http::trace::on_request", "my_crate::db", "warp::filters::log", "tracing::span",
    "rustls::client", "h2::codec", "mio::poll", "app::worker::scheduler", "axum::rejection",
]
NODE_MODULES = [
    "app", "server", "http", "express:router", "app:db", "worker", "api", "auth", "mongoose", "socket.io",
    "next", "vite", "webpack", "prisma:query", "koa", "nest", "NestApplication", "RouterExplorer",
    "InstanceLoader", "TypeOrmModule", "app:server", "app:cache", "queue:jobs", "mailer",
]
USERS = ["alice", "bob", "root", "admin", "jdoe", "svc-deploy", "www-data", "postgres", "ubuntu",
         "carol.smith", "deploy", "guest", "webmaster", "test", "oracle", "m.mueller", "user_42", "ops"]
SERVICES = ["auth-service", "payments", "user-api", "gateway", "orders", "billing", "search", "notifier",
            "checkout", "inventory", "scheduler", "ingest", "frontend", "backend", "worker", "cron-runner",
            "recommendations", "ledger", "webhooks", "mailer", "cache-warmer", "sync"]
PATHS = ["/", "/api/v1/users", "/api/v1/users/123", "/health", "/healthz", "/metrics", "/login",
         "/static/app.js", "/favicon.ico", "/api/orders?page=2&limit=50", "/v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail",
         "/wp-admin/admin-ajax.php", "/images/logo.png", "/api/search?q=hello%20world", "/graphql",
         "/robots.txt", "/.env", "/admin", "/api/v2/items/9f8e7d6c", "/ws", "/oauth/callback?code=abc123&state=xyz",
         "/products/42/reviews", "/index.html", "/assets/main.css", "/api/v1/auth/refresh"]
METHODS = ["GET", "GET", "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
STATUSES = ["200", "200", "200", "201", "204", "301", "302", "304", "400", "401", "403", "404", "404",
            "409", "422", "429", "500", "502", "503", "504"]
UAS = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
       "curl/8.4.0", "python-requests/2.31.0", "Go-http-client/1.1", "kube-probe/1.28", "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
       "PostmanRuntime/7.36.0", "Googlebot/2.1 (+http://www.google.com/bot.html)", "-", "okhttp/4.12.0",
       "ELB-HealthChecker/2.0", "axios/1.6.2", "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"]
REFERERS = ["-", "-", "https://example.com/", "https://www.google.com/", "https://example.com/products",
            "http://localhost:3000/", "https://app.example.com/dashboard", "android-app://com.example.app"]
FILES_JAVA = ["Foo.java", "UserService.java", "Thread.java", "HttpClient.java", "Main.java", "Dispatcher.java",
              "AbstractHandler.java", "NativeMethodAccessorImpl.java", "Method.java", "ReflectiveOperationException.java",
              "Kt.kt", "Application.kt", "Routes.scala", "ForkJoinTask.java", "Executors.java", "SocketInputStream.java"]
JAVA_CLASSES = ["com.example.api.UserController", "com.example.service.UserService", "java.base/java.lang.Thread",
                "org.springframework.web.servlet.DispatcherServlet", "java.net.SocketInputStream",
                "jdk.internal.reflect.NativeMethodAccessorImpl", "java.util.concurrent.ThreadPoolExecutor$Worker",
                "org.apache.catalina.core.ApplicationFilterChain", "com.acme.billing.InvoiceService",
                "sun.reflect.DelegatingMethodAccessorImpl", "org.hibernate.engine.jdbc.internal.StatementPreparerImpl",
                "io.netty.channel.AbstractChannelHandlerContext", "com.zaxxer.hikari.pool.PoolBase",
                "kotlin.coroutines.jvm.internal.BaseContinuationImpl", "org.junit.runners.ParentRunner$3",
                "com.example.Main", "scala.concurrent.impl.Promise$Transformation", "java.base/java.util.concurrent.FutureTask"]
JAVA_METHODS = ["run", "invoke", "doFilter", "handle", "execute", "call", "process", "service", "getUser", "save",
                "flush", "read", "socketRead0", "invoke0", "lambda$main$0", "<init>", "<clinit>", "apply", "accept",
                "doGet", "doPost", "processRequest", "resumeWith", "runWorker", "invokeSuspend", "access$100"]
JAVA_EXC = ["java.lang.NullPointerException", "java.io.IOException", "java.lang.IllegalStateException",
            "java.net.SocketTimeoutException", "org.springframework.beans.factory.BeanCreationException",
            "java.sql.SQLException", "java.lang.OutOfMemoryError", "javax.persistence.PersistenceException",
            "java.util.concurrent.TimeoutException", "com.example.NotFoundException", "kotlin.KotlinNullPointerException",
            "java.lang.RuntimeException", "java.lang.ClassNotFoundException", "org.apache.kafka.common.errors.TimeoutException"]
PY_FILES = ["/usr/lib/python3.12/site-packages/requests/adapters.py", "/app/main.py", "/app/services/payment.py",
            "/home/alice/project/venv/lib/python3.11/site-packages/django/core/handlers/base.py",
            "/usr/local/lib/python3.10/asyncio/events.py", "./scripts/run.py", "app/api/routes.py",
            "/opt/airflow/dags/etl_pipeline.py", "C:\\Users\\jdoe\\project\\src\\utils.py", "<stdin>", "<string>",
            "/usr/lib/python3/dist-packages/urllib3/connectionpool.py", "/app/worker/tasks.py",
            "/src/pipeline/transform.py", "manage.py", "/usr/local/lib/python3.12/threading.py"]
PY_FUNCS = ["main", "<module>", "run", "send", "_make_request", "handle", "process_item", "get_response",
            "inner", "wrapper", "_run_once", "run_forever", "connect", "fetch", "execute", "_bootstrap_inner",
            "__call__", "dispatch", "load", "parse_args", "__init__", "on_message", "transform"]
PY_EXC = ["ValueError", "KeyError", "TypeError", "AttributeError", "requests.exceptions.ConnectionError",
          "ZeroDivisionError", "FileNotFoundError", "RuntimeError", "IndexError", "json.decoder.JSONDecodeError",
          "psycopg2.OperationalError", "asyncio.exceptions.TimeoutError", "ModuleNotFoundError", "PermissionError",
          "sqlalchemy.exc.OperationalError", "django.db.utils.IntegrityError", "botocore.exceptions.ClientError"]
JS_FILES = ["/app/src/server.js", "/app/node_modules/express/lib/router/index.js", "/home/alice/project/dist/index.js",
            "node:internal/process/task_queues", "node:internal/modules/cjs/loader", "/app/src/routes/users.ts",
            "webpack-internal:///./src/App.tsx", "https://cdn.example.com/static/js/main.4f2a1c.js",
            "http://localhost:3000/static/js/bundle.js", "/usr/lib/node_modules/npm/lib/cli.js", "file:///app/dist/main.mjs",
            "/app/node_modules/mongoose/lib/model.js", "src/components/Button.jsx", "C:\\Users\\jdoe\\app\\index.js",
            "/app/.next/server/pages/api/hello.js", "node:events", "/app/node_modules/pg/lib/client.js"]
JS_FUNCS = ["Object.<anonymous>", "Module._compile", "processTicksAndRejections", "Layer.handle [as handle_request]",
            "next", "Router.handle", "async handler", "new Promise (<anonymous>)", "Promise.then", "Socket.emit",
            "ClientRequest.<anonymous>", "Timeout._onTimeout", "listOnTimeout", "process.processTimers", "fetchUser",
            "main", "app.get", "Function.Module._load", "TCPConnectWrap.afterConnect [as oncomplete]", "emitErrorNT",
            "Server.<anonymous>", "async Promise.all (index 0)", "Array.forEach (<anonymous>)", "Object.handle"]
JS_EXC = ["TypeError", "ReferenceError", "Error", "RangeError", "SyntaxError", "UnhandledPromiseRejection",
          "MongoServerError", "AxiosError", "ValidationError", "PrismaClientKnownRequestError", "FetchError"]
GO_FILES = ["/home/alice/go/src/github.com/acme/app/main.go", "/app/pkg/server/handler.go", "/usr/local/go/src/runtime/panic.go",
            "/usr/local/go/src/net/http/server.go", "/go/pkg/mod/github.com/gin-gonic/gin@v1.9.1/context.go",
            "/usr/local/go/src/runtime/proc.go", "/build/internal/db/pool.go", "/usr/local/go/src/runtime/asm_amd64.s",
            "/go/src/app/cmd/agent/main.go", "/root/go/pkg/mod/google.golang.org/grpc@v1.59.0/server.go", "main.go",
            "handler.go", "pkg/api/users.go", "/app/controllers/reconcile.go"]
GO_FUNCS = ["main.main", "main.(*Server).handle", "net/http.(*conn).serve", "net/http.HandlerFunc.ServeHTTP",
            "github.com/acme/app/pkg/server.(*Handler).ServeHTTP", "runtime.goexit", "runtime.gopanic",
            "github.com/gin-gonic/gin.(*Context).Next", "main.worker", "sync.(*WaitGroup).Wait", "runtime.main",
            "google.golang.org/grpc.(*Server).serveStreams.func1.2", "main.run.func1", "internal/db.(*Pool).Query",
            "k8s.io/client-go/tools/cache.(*controller).Run", "runtime.panicIndex", "main.(*App).Start"]
RUST_FILES = ["/rustc/79e9716c980570bfd1f666e3b16ac583f0168962/library/std/src/panicking.rs", "src/main.rs",
              "/home/alice/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.35.0/src/runtime/task/core.rs",
              "/rustc/79e9716c980570bfd1f666e3b16ac583f0168962/library/core/src/panicking.rs", "src/server/mod.rs",
              "/rustc/79e9716c980570bfd1f666e3b16ac583f0168962/library/core/src/option.rs", "crates/app/src/db.rs",
              "/usr/src/app/src/handlers/users.rs", "src/lib.rs", "/rustc/a28077b28a02b92985b3a3faecf92813155f1ea1/library/std/src/rt.rs"]
RUST_FUNCS = ["std::panicking::begin_panic", "core::panicking::panic_fmt", "app::main", "core::option::Option<T>::unwrap",
              "std::rt::lang_start::{{closure}}", "tokio::runtime::task::core::Core<T,S>::poll", "app::server::run",
              "<core::future::from_generator::GenFuture<T> as core::future::Future>::poll", "rust_begin_unwind",
              "std::panicking::rust_panic_with_hook", "app::handlers::users::get_user::{{closure}}", "core::result::unwrap_failed",
              "std::sys_common::backtrace::__rust_end_short_backtrace", "main", "__libc_start_main", "_start"]
CS_FILES = ["C:\\src\\App\\Controllers\\UserController.cs", "/src/App/Services/PaymentService.cs",
            "/_/src/libraries/System.Private.CoreLib/src/System/Threading/Tasks/Task.cs", "D:\\a\\1\\s\\src\\Program.cs",
            "C:\\BuildAgent\\work\\app\\Data\\Repository.cs", "/app/src/Worker.cs", "/home/alice/app/Startup.cs"]
CS_FUNCS = ["App.Controllers.UserController.Get(Int32 id)", "System.Threading.Tasks.Task.ThrowIfExceptional(Boolean includeTaskCanceledExceptions)",
            "App.Services.PaymentService.<ChargeAsync>d__12.MoveNext()", "Microsoft.AspNetCore.Mvc.Infrastructure.ControllerActionInvoker.<InvokeActionMethodAsync>g__Awaited|12_0(ControllerActionInvoker invoker, ValueTask`1 actionResultValueTask)",
            "System.Runtime.CompilerServices.TaskAwaiter.HandleNonSuccessAndDebuggerNotification(Task task)", "Program.Main(String[] args)",
            "App.Data.Repository`1.Find(Guid id)", "Npgsql.NpgsqlConnector.<Open>d__57.MoveNext()", "App.Worker.ExecuteAsync(CancellationToken stoppingToken)",
            "lambda_method(Closure , Object , Object[] )", "System.Collections.Generic.Dictionary`2.get_Item(TKey key)"]
CS_EXC = ["System.NullReferenceException", "System.InvalidOperationException", "System.ArgumentNullException",
          "System.IO.FileNotFoundException", "Npgsql.PostgresException", "System.Net.Http.HttpRequestException",
          "System.AggregateException", "Microsoft.EntityFrameworkCore.DbUpdateException", "System.TimeoutException"]
RB_FILES = ["/app/app/controllers/users_controller.rb", "/usr/local/bundle/gems/actionpack-7.1.2/lib/action_controller/metal/basic_implicit_render.rb",
            "/usr/local/bundle/gems/activerecord-7.1.2/lib/active_record/connection_adapters/abstract_adapter.rb",
            "/app/lib/tasks/import.rake", "/usr/local/bundle/gems/puma-6.4.0/lib/puma/thread_pool.rb", "app.rb", "script.rb",
            "/home/alice/.rbenv/versions/3.2.2/lib/ruby/3.2.0/net/http.rb", "/app/config/initializers/redis.rb", "lib/worker.rb"]
RB_FUNCS = ["show", "send_action", "block in process_action", "each", "<main>", "block in <main>", "call", "run",
            "perform", "execute", "connect", "block (2 levels) in <class:Worker>", "with_connection", "spawn_thread",
            "rescue in transaction", "load", "require", "block in spawn_thread", "<top (required)>", "Foo#bar", "Integer#+"]
RB_EXC = ["NoMethodError", "ArgumentError", "RuntimeError", "ActiveRecord::RecordNotFound", "Errno::ECONNREFUSED",
          "NameError", "ZeroDivisionError", "Redis::CannotConnectError", "JSON::ParserError", "ActionController::RoutingError"]
PHP_FILES = ["/var/www/html/index.php", "/var/www/app/vendor/laravel/framework/src/Illuminate/Routing/Router.php",
             "/var/www/html/wp-content/plugins/woocommerce/includes/class-wc-cart.php", "/app/src/Controller/UserController.php",
             "/var/www/app/vendor/symfony/http-kernel/HttpKernel.php", "/srv/app/public/index.php", "/var/www/html/wp-includes/plugin.php",
             "/app/vendor/doctrine/dbal/src/Connection.php", "/var/www/app/artisan", "/usr/share/php/PHPUnit/Framework/TestCase.php"]
PHP_FUNCS = ["App\\Http\\Controllers\\UserController->show()", "Illuminate\\Routing\\Router->dispatch()",
             "WC_Cart->calculate_totals()", "Symfony\\Component\\HttpKernel\\HttpKernel->handleRaw()", "require_once()",
             "do_action()", "Doctrine\\DBAL\\Connection->executeQuery()", "App\\Kernel->handle()", "call_user_func_array()",
             "PDO->__construct()", "Illuminate\\Foundation\\Application->run()", "array_map()", "App\\Jobs\\SendEmail->handle()"]
PHP_EXC = ["Exception", "ErrorException", "PDOException", "Illuminate\\Database\\QueryException", "TypeError",
           "Symfony\\Component\\HttpKernel\\Exception\\NotFoundHttpException", "RuntimeException", "InvalidArgumentException"]
K8S_NS = ["default", "kube-system", "prod", "staging", "monitoring", "ingress-nginx", "cert-manager", "argocd"]
PODS = ["api-7d9f8b6c4-x2k9p", "web-deployment-5f6d7c8b9-abcde", "nginx-ingress-controller-abc12", "coredns-5d78c9869d-8mzgh",
        "worker-0", "redis-master-0", "postgres-1", "prometheus-server-6b7f8c9d-qwert", "cert-manager-webhook-7c8d9e-zxcvb"]
ERRORS = ["connection refused", "connection reset by peer", "timeout exceeded", "no such file or directory",
          "permission denied", "invalid token", "unexpected EOF", "context deadline exceeded", "too many open files",
          "broken pipe", "host unreachable", "i/o timeout", "duplicate key value violates unique constraint",
          "record not found", "out of memory", "certificate has expired", "dial tcp 10.0.0.5:5432: connect: connection refused",
          "EOF", "rate limit exceeded", "invalid JSON payload", "TLS handshake timeout", "address already in use"]
